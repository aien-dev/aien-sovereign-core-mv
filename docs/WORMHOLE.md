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
- `wormhole_run ring-check` and `wormhole_run pair` (the real-weight exactness tests) are written and
  queued; their results land in the receipt README.

### 9.2 Acceptance so far (code-golden, 30 words; prose and codelong queued)

Flash-Next drafting for the 27B agreed with the 27B on 29 of 30 words. Words delivered per round:
k=1 2.00, k=2 2.73, k=4 4.29, k=8 7.50, k=16 15.0. Expert LRU hit on that trace: 4 GB 0.60, 8 GB 0.72,
16 GB 0.80. This prompt is easy code; the longer prose and code sets decide it.

### 9.3 Break-even and ceiling from those numbers

Floors from section 6 (Flash-Next on the chip at the same 273 GB/s as the 27B, misses not hidden):

| cache | Flash-Next word | k=1 | k=2 | k=4 | k=8 | k=16 | 27B alone |
|---|---|---|---|---|---|---|---|
| perfect (all experts in memory) | 36.4 ms | 14.8 | 15.9 | 17.6 | 19.3 | 22.0 tok/s | 10.15 |
| 8 GB, hit 0.72 | 73.3 ms | 11.6 | 11.1 | 10.9 | 11.0 | 11.8 tok/s | 10.15 |
| 4 GB, hit 0.60 | 89.7 ms | 10.6 | 9.8 | 9.4 | 9.2 | 9.8 tok/s | 10.15 |

Plainly: on this one easy prompt the pairing only just beats the 27B ceiling with an 8 GB cache, and loses
with 4 GB. The SSD misses eat almost all of the gain. The whole lever is the cache.

### 9.4 What would make it win (the next tests)

1. Hide the misses behind the verify pass. While the 27B verifies a round (98.5 ms), the SSD can deliver
   0.99 GB = about 357 experts. That only helps if the next round's experts can be predicted. Measured on
   the code-golden trace: the experts the previous word used in the same layer cover 0.366 of the next
   word's experts; the last 2, 4 and 8 words cover 0.490, 0.572 and 0.646. Routing is not sticky enough
   for history alone to hide the misses. The next test is look-ahead routing: apply layer L+1's router to
   layer L's output (the hidden state changes little between layers) and measure how many of layer L+1's
   real experts it names, on the CPU reference; if that is high, the ring can fetch layer L+1's experts
   while layer L computes. A limit to know before building it: one Flash-Next layer computes in about
   0.76 ms at the memory floor (36.4 ms / 48), while 10 cold experts take 2.76 ms from the SSD. So even a
   perfect one-layer look-ahead hides at most about a quarter of a full miss; it has to predict several
   layers ahead, or hide the misses inside the 27B verify pass, to matter.
2. A deeper cache: the 16 GB row (hit 0.80 even on a cold 30-word trace) needs tenants to give back memory.
3. Pick k per measured acceptance. At 8 GB the table is flat (10.9 to 11.8 tok/s for k = 1 to 16): each
   drafted word costs most of what it saves, so k only matters once misses are hidden.
4. The Flash-Next chip forward (L020) has to exist before any of this can be timed.

### 9.5 Race

| racer | tok/s | ceiling | status |
|---|---|---|---|
| 27B alone (shared baseline, not re-measured) | 7.63 live | 10.15 | receipt from serve-fast |
| 27B + DFlash (L019) | none | not computed (no acceptance yet) | no run: L019's chip runs were refused for memory (36.8 GB needed with 12 GB kept free) and failed a channel alloc; not run here by agreement with L019 |
| 27B verifying, Flash-Next drafting | none | section 9.3 | cannot run: there is no Flash-Next chip forward yet (L020 kernels are simulation proven only) |
| Flash-Next verifying, 27B drafting | none | below Flash-Next alone at any acceptance (section 6) | cannot run, same reason; physically cannot win |

No timed race was run. The result for the race is INCONCLUSIVE, not a pass.
