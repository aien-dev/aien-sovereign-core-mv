# Wormhole: Qwen3.8-27B and Qwen3.8-Flash-Next as draft and block verify

Ledger L080. Branch `wormhole/27b-flashnext` of aien-sovereign-core-mv. Status and numbers: section 9.

## 1. The idea in one paragraph

Two models that share one vocabulary are joined. One of them (the drafter) guesses the next few words
cheaply. The other (the verifier) checks all of those guesses in a single pass, keeps the longest run of
guesses that match what it would have written itself, and adds its own next word. The 27B reads all of its
weights once per pass, whether the pass checks one word or seventeen, so a round that is accepted often
costs about one 27B word but delivers several. Flash-Next is too big to sit in memory beside the 27B
(124 GB on disk, 68 GB of it routed experts), so its experts live on the SSD and come in through the Expert
Heartbeat ring (aienix#4): fixed slots, an expert is readable only after its whole read finished and its
BLAKE3 digest matched the manifest.

## 2. Why these two models, and not the 30B

| model | vocabulary (tokenizer) | tokenizer.json sha256 prefix |
|---|---|---|
| qwen3-30b-a3b-instruct-2507-fp8 (the heartbeat's first MoE) | 151,643 (vocab_size 151,936) | different |
| qwen3.8-27b-fp8-017b9c7a | 248,044 (vocab_size 248,320) | 0997f410c57a1f4e |
| qwen3.8-flash-next-nvfp4-fc694b54 | 248,044 (vocab_size 248,320) | 0997f410c57a1f4e |

Word by word drafting needs the same vocabulary, so the 30B is out. Flash-Next and the 27B have byte
identical tokenizers.

## 3. Roles

- Forward (the product): Flash-Next drafts, the 27B verifies. The output is the 27B's own greedy output.
- Reversed (measured for completeness): the 27B drafts, Flash-Next verifies. The output is then
  Flash-Next's own greedy output, a different product. A drafter that reads more bytes per word than the
  verifier can never beat the verifier alone (section 6), so this pairing is a control, not a candidate.
- DFlash drafter (L019, qwen3.8-27b-dflash2, 3.6 GB) drafting for the 27B: the third racer, owned by L019.

## 4. The exactness rule and its test

The Wormhole's output must be exactly, word for word, what the verifier alone writes with greedy
decoding. The drafter can only change speed, never text.

How the round keeps the rule (crates/aien-wormhole/src/spec.rs):

1. The verifier holds a committed state and one pending word (its own choice, already decided).
2. The drafter, from its committed state, drafts up to k words after the pending word.
3. The verifier feeds `[pending, d1..dk]` in one pass and gets its own greedy choice after every one.
4. Accept `d1..dm` while each equals the verifier's choice at that position. The next pending word is the
   verifier's choice after `dm` (its correction, or the bonus word when all k matched).
5. Both models commit exactly `pending, d1..dm`; anything past that is thrown away (the chip path does
   this by snapshot and rollback of the DeltaNet and conv states, which the 27B MTP work already proved
   exact; the host code re-feeds from the last commit).

Every word written is a verifier choice made on the verifier's own committed prefix, so the text cannot
differ from the verifier alone.

Tests (all must pass; the controls must fail):

| test | what it shows | result |
|---|---|---|
| `spec::tests::output_equals_verifier_alone_for_every_block_size` | toy causal models, drafters wrong on every 2nd/3rd/5th context or never, k = 1..16: output == verifier alone | pass |
| `spec::tests::negative_controls_go_red` | accept one refused draft; compare against the wrong row: output must differ | both differ (controls work) |
| `wormhole_run pair` | the same round with the real weights on the CPU references: Flash-Next (experts through the ring) drafts, the 27B verifies, output == 27B alone; `WORMHOLE_FAULT=extra` must differ | section 9 |
| `wormhole_run ring-check` | Flash-Next with its experts read through the Heartbeat ring, one word at a time, vs the plain checkpoint reader on the whole prompt: every logit bit identical; a replaced digest must fail closed | section 9 |

## 5. Memory budget (27B resident)

Bytes from the safetensors headers of both checkpoints (`bytes.py`, receipt in the evidence folder).

| piece | bytes | where it lives |
|---|---|---|
| 27B weights read every pass (all but embedding, vision, MTP) | 26.93 GB | resident |
| 27B embedding (row gather), MTP head, vision | 2.54 + 0.48 + 0.92 GB | resident or skipped |
| Flash-Next non-expert weights read every word: Gated DeltaNet 4.17, attention 1.23, hyper connections 1.28, lm_head 1.27, shared expert 0.47, router 0.13, other 0.07 | 8.62 GB | resident |
| Flash-Next embedding (row gather) | 1.27 GB | resident |
| Flash-Next n-gram table (16 row lookups per word) | 51.20 GB | SSD, row reads (a few KB per word) |
| Flash-Next MTP head 2.70, vision 0.90 | 3.60 GB | not loaded |
| Flash-Next routed experts: 48 layers x 512 experts, 2,764,824 bytes each | 67.95 GB | SSD, Heartbeat ring cache |

Machine: 121 GiB (about 130 GB). Tenants (three aien-serve-http, agents, builds) use most of it: at the
time of writing `free -g` showed 14 to 18 GiB available. With the 27B already loaded by the serving stack,
the Wormhole adds about 9.9 GB of Flash-Next core plus the expert cache. So the cache that really fits
today is a few GB (about 1,500 to 3,000 slots of 24,576, 6 to 12 percent of all experts). Hit rates for
caches of 2, 4, 8, 16 and 32 GB are measured in section 9.

## 6. The physics (bytes per round over measured bandwidth)

Measured inputs:
- 27B ceiling 10.15 tok/s (shared baseline, calc:9822c2ba) over 26.93 GB per pass = 273 GB/s effective
  memory read. One 27B pass: 98.5 ms at the ceiling (live today 7.63 tok/s = 131 ms).
- SSD: 10.03 GB/s (heartbeat H0 receipt).
- Flash-Next word, all experts in memory: 8.62 + 1.33 GB = 9.95 GB = 36.4 ms at 273 GB/s.
- Flash-Next word, every expert cold: 480 experts x 2.76 MB = 1.327 GB from the SSD = 132 ms extra
  (more than a 27B pass at the ceiling). With miss rate m the drafter word costs 36.4 + 132 m ms if the
  read cannot hide (the router picks layer L's experts only after layer L-1, so a miss stalls).

Forward round, k drafts, E words delivered per round (E from the acceptance receipt):

    T(k) = k x (36.4 + 132 m) ms + 98.5 ms        speed = E / T(k)        27B alone = 1 / 98.5 ms

Break-even: E > T(k) / 98.5 ms. With a perfect cache (m = 0): k=4 needs E > 2.48, k=8 needs E > 3.96,
k=16 needs E > 6.91. With m = 0.5, k=4 needs E > 5.16, which is more than the 5 words a k=4 round can give:
impossible.

Reversed round: every draft costs a whole 27B pass (98.5 ms) and Flash-Next alone costs 36.4 + 132 m ms per
word, so the reversed pair is slower than Flash-Next alone at any acceptance.

## 7. Ring rhythm and the expert cache

- Slots: one expert per slot, slot size = the largest expert (2,764,824 bytes), one manifest of all 24,576
  experts (byte spans in the shards plus BLAKE3), built once (`wormhole_run manifest`).
- Beat: one beat per drafted word. Before the word, `prefetch` fills (on 8 reader threads, one writer per
  slot) the experts predicted for it; during the word, any miss is a demand fill into the least recently
  used slot. A slot is published only after digest and fence (aien-heartbeat `Ring`), a reader takes a lease,
  and a writer waits for leases to drop before reusing a slot. Nothing half written is ever read.
- Prediction for the prefetch: the experts the previous word used in the same layer (routing is sticky from
  word to word). The trace measurement in section 9 gives the hit rate of LRU and of a pinned hot set.
- The verifier (27B) never waits on the SSD: it is fully resident.

## 8. The verify path on the chip

- 27B block verify: the L013 tree-verify GEMM (bits identical to the one-row path for every q38 shape,
  head 2a0484d1) runs the k+1 rows of a round as one weight read.
- State rollback: snapshot of DeltaNet and conv states before the round, as in the 27B MTP path
  (qwen38-spec evidence, exact for k = 1 to 4).
- Flash-Next on the chip: the forward is L020's port (flashnext/port). Its new kernels are simulation proven
  only (evidence flashnext-m3-sim-20261019T2010Z); there is no chip Flash-Next forward yet.

## 9. Status and numbers

Receipts: `evidence/WORMHOLE-ACCEPT-69825c92/README.md` (tables are appended there as runs land).

### 9.1 What is built and passing

- `crates/aien-wormhole`: the expert cache on the Heartbeat ring (`cache.rs`: versioned slots, demand fill
  into the least recently used slot, parallel `prefetch`, fail closed on a digest mismatch), the tensor
  source the Flash-Next reference reads its routed experts through (`source.rs`), and the round
  (`spec.rs`). Unit tests pass: exactness against the verifier alone for k = 1 to 16 with four drafter
  qualities, and both negative controls change the output as they must.
- `aien-heartbeat`: the manifest now reads Flash-Next's `model.language_model.layers.*` expert names and
  digests its 24,576 experts on 8 threads; `Streamer::load_one` is public for demand fill. Its 9 + 8 tests
  pass (three runs; `writer_waits_for_lease_release` failed once under load 37 because it relies on a
  10 ms sleep, which is a test timing assumption, not a ring fault; it passed the next three times).
- `wormhole_run ring-check` (Flash-Next's real experts through the ring, bit for bit, plus a corrupted
  digest that must fail closed) and `wormhole_run pair` (real weights: Flash-Next drafts through the ring,
  the 27B verifies, output must equal the 27B alone; and the accept-one-extra control) are written; their
  runs are chained after the acceptance runs and land in the receipt README. Until then they are not results.

### 9.2 Acceptance: Flash-Next drafting, 27B verifying (all three prompt sets)

Verifier text = the 27B's own chip greedy output. Cell = accepted / drafted, then words delivered per round.

| prompt set | words | Flash-Next top-1 = 27B word | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|---|
| code-golden | 30 | 29 / 30 | 15/15, 2.00 | 20/22, 2.73 | 24/28, 4.29 | 27/29, 7.50 | 29/30, 15.0 |
| prose | 256 | 181 / 256 | 108/148, 1.73 | 143/226, 2.27 | 166/358, 2.84 | 178/613, 3.28 | 181/1159, 3.41 |
| codelong | 256 | 227 / 256 | 119/138, 1.86 | 158/198, 2.59 | 192/256, 4.00 | 210/371, 5.45 | 222/548, 7.31 |

For scale, the 27B's own MTP head on the same codelong text (qwen38-spec evidence): k=1 120/136,
k=4 185/288. Flash-Next guesses about as well as the 27B's own small MTP head, and the MTP head costs about
0.48 GB per guess against Flash-Next's 9.95 GB plus SSD misses.

### 9.3 Expert cache on the real traces

All three traces (542 words of 27B text, 480 expert lookups per word), LRU with each prompt warming it:

| cache | experts | share of 24,576 | LRU hit | hot set chosen on the other prompts |
|---|---|---|---|---|
| 2 GB | 723 | 2.9% | 0.451 | 0.015 |
| 4 GB | 1,446 | 5.9% | 0.622 | 0.061 |
| 8 GB | 2,893 | 11.8% | 0.765 | 0.145 |
| 16 GB | 5,786 | 23.5% | 0.901 | 0.295 |
| 32 GB | 11,573 | 47.1% | 0.952 | 0.537 |

Two findings. There are no globally hot experts: a set pinned from other prompts hits about as often as
chance, so the cache must follow the current text (LRU), not a fixed favourite list. And routing is only
partly sticky from word to word: the previous word's experts in the same layer cover 0.37 to 0.44 of the
next word's, the last 8 words 0.65 to 0.69.

### 9.4 Break-even and ceiling (bytes per round over measured bandwidth)

Section 6 formula, Flash-Next assumed to run on the chip at the 27B's 273 GB/s, misses not hidden. tok/s at
the best k for each cache; 27B alone is 10.15 at the same ceiling.

| cache (hit) | Flash-Next word | code-golden | prose | codelong |
|---|---|---|---|---|
| perfect (1.000) | 36.4 ms | 22.0 (k=16) | 13.3 (k=2) | 16.4 (k=4) |
| 32 GB (0.952) | 42.7 ms | 19.2 (k=16) | 12.3 (k=2) | 14.8 (k=4) |
| 16 GB (0.901) | 49.5 ms | 16.8 (k=16) | 11.7 (k=1) | 13.5 (k=4) |
| 8 GB (0.765) | 67.5 ms | 12.7 (k=16) | 10.4 (k=1) | 11.2 (k=1) |
| 4 GB (0.622) | 86.4 ms | 10.8 (k=1) | 9.4 (k=1) | 10.1 (k=1) |

Hit rate needed just to tie the 27B alone: prose 0.73 at k=1 (0.93 at k=4), codelong 0.64 at k=1
(0.72 at k=4). The cache that fits beside the 27B today is about 4 GB (hit 0.62): below break-even on
every long set. Even with all 68 GB of experts in memory, which cannot fit, the best case is 13.3 tok/s on
prose and 16.4 on codelong.

Against the 27B's own MTP head on the same formula (0.48 GB per guess, 1.75 ms; codelong k=1 delivers
256 words in 136 rounds = 1.88 per round): 1.88 / (98.5 + 1.75) ms = 18.8 tok/s ceiling. That beats every
Flash-Next row, including the impossible perfect cache. The plain answer: as a drafter for the 27B,
Flash-Next cannot beat what the 27B already has.

Reversed (27B drafting, Flash-Next verifying): each guess costs a whole 27B pass (98.5 ms) while Flash-Next
alone costs 36.4 to 86 ms a word, so it is slower than Flash-Next alone at any acceptance. Its acceptance is
still measured (queued) for the record.

### 9.5 What would make it win (tested so far, and next)

1. Hide misses behind the verify pass. While the 27B verifies (98.5 ms) the SSD delivers 0.99 GB, about
   357 experts; that helps only if the next word's experts can be predicted. History predicts 0.37 to 0.69
   (above), not enough. One Flash-Next layer computes in 0.76 ms at the floor while its 10 cold experts take
   2.76 ms from the SSD, so even perfect one-layer look-ahead routing hides at most a quarter of a miss.
   Untested: look-ahead several layers deep (layer L+n's router on layer L's output) on the CPU reference.
2. A deeper cache: 16 GB (hit 0.90) gets prose to 11.7 and codelong to 13.5, still under the MTP head's
   18.8 ceiling, and 16 GB is not free on this box under tenant load.
3. Smaller block: already in the table; k=1 or 2 is best whenever misses are real.
4. Use the pair the other way round in role, not in output: Flash-Next as the product, with its own small
   drafter (its MTP head, 2.7 GB, or the 27B's text as a hint) is the only arrangement where Flash-Next's
   cost is not pure overhead. That is a different product (Flash-Next's text) and needs L020's chip forward.
5. The real lever for the 27B is cheap drafting (its MTP head now, DFlash L019) plus the L013 block verify,
   not a second large model.

### 9.6 Race

| racer | tok/s | ceiling | status |
|---|---|---|---|
| 27B alone (shared baseline, not re-measured) | 7.63 live | 10.15 | receipt from serve-fast |
| 27B + DFlash (L019) | none | not computed (no acceptance yet) | no run: L019's chip runs were refused for memory (36.8 GB needed with 12 GB kept free) and failed a channel alloc; not run here by agreement with L019 |
| 27B verifying, Flash-Next drafting | none | 9.4 to 16.4 by cache and text, 10.1 to 10.8 with the 4 GB that fits (section 9.4) | cannot run: there is no Flash-Next chip forward yet (L020 kernels are simulation proven only) |
| Flash-Next verifying, 27B drafting | none | below Flash-Next alone at any acceptance (section 6) | cannot run, same reason; physically cannot win |

No timed race was run. The result for the race is INCONCLUSIVE, not a pass.
