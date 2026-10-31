# Q36 host reference oracle (L306, 2026-10-20)

Our own Rust CPU forward pass (`qwen38_check`, crate aien-inference-abi) on Qwen3.6-35B-A3B-FP8 rev 95a723d0.
Teacher-forced score of the first 400 tokens of the AGPL-3.0 text (`passage-agpl.txt`, a text the model has seen
many times, so a correct forward pass predicts it almost perfectly):

| routing                                      | mean NLL | perplexity | file                     |
|----------------------------------------------|----------|------------|--------------------------|
| correct (softmax over 256, top-8, renorm)    | 0.164    | 1.18       | score-agpl.json          |
| mutant 1: 8th pick swapped for the 9th       | 0.095    | 1.10       | score-agpl-mutant.json   |
| mutant 2: every pick sent to expert id + 1   | 4.594    | 98.85      | score-agpl-mutant2.json  |

Reading: mutant 2 (wrong experts) is red by 84x in perplexity, so the expert wiring is checked. Mutant 1 is NOT a
useful oracle at the language level: the 8th and 9th experts are near-ties with small renormalised weight, and the
swap moved the score slightly the other way. It stays as a bit-exact negative control (chip ids vs host ids), never
as a quality signal. README passage (`score-readme.json`, ppl 17.5) is a weak oracle (project-specific text).
