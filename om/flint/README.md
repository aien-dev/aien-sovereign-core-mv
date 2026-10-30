# flint: pinned checkpoint manifest reader (Core #802 P0)

`manifest.om` is the first deliverable of Core #802 (aien-architecture-mv#353): a bounded
reader that admits the Flint BF16 checkpoint at one pinned revision and refuses everything
else. Memory, kernel, training and serving work read its output, the tensor inventory TSV,
instead of parsing safetensors themselves. It does not execute the model.

Pinned source: `RecursiveSI/flint` revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`,
Apache-2.0 (see `THIRD_PARTY.md`).

| File | sha256 |
|---|---|
| `config.json` (2451 bytes, copied to `test/config.json`) | `a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7` |
| `model.safetensors.index.json` (3329405 bytes) | `e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311` |
| inventory TSV printed by the reader (36734 rows plus the header line) | `60ff4dbdaa1c3632c934c6935aa60ed5b2def4e15f0ca5083ddc7dd660f083f5` |

Build with omega-mv `bf54b19fa590e8ae925c0e80815516dbf41e7326`, seeded omc sha256
`62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`. `om/vendor/json.om` is
byte-identical to that revision's `seed/json/json.om` (sha256 `456254fc…e2d2`). This is a newer
compiler pin than `om/tur-ledger` (`6999fa9e`, omc `d6290c76…`).

```sh
{ printf 'om2\n'; cat om/vendor/json.om om/flint/manifest.om; } | "$OMC" > flint-manifest && chmod +x flint-manifest
./flint-manifest CONFIG INDEX SHARD_DIR > inventory.tsv
```

On success the exit status is 0, stdout is the TSV and stderr is one line:

```
flint-manifest ok tensors 36734 params 39785287456 routed 37849399296 shards 15 bytes 79570574912
```

TSV columns: `name shard dtype shape begin end`. Shards come in name order; within a shard,
rows follow the order of the header's entries. `begin`/`end` are byte offsets into the shard's data section,
which starts at `8 + header length`. `shape` is `d0xd1` or `d0`.

## What it checks

- **Config.** Every key of the pinned `config.json` is recognised or the file is refused, and
  so is a duplicate key. Every value that affects the forward path is pinned. Examples:
  architecture `FlintForCausalLM`; 49 layers; hidden 2048, head_dim 128, 32 query and 4 KV
  heads; 256 experts, top-6, `norm_topk_prob` true; `router_score_function` `sigmoid`;
  `softmax_type` `learnable`; `sliding_window` 513; the full-attention set {0, 1, 5, 9, ..., 45, 48},
  which must equal `no_rope_layer_indices`; `mlp_only_layers` [0, 1]; untied embeddings; vocab
  150144; `rms_norm_eps` 1e-6 and `rope_theta` 10000, compared as exact decimals; `rope_scaling`
  null. `attention_dropout`, `initializer_range`, `output_router_logits`,
  `router_aux_loss_coef`, `use_cache` and `transformers_version` may be absent; when present
  only their type is checked.
- **Index.** The index is read with a strict JSON subset scanner: ASCII without escapes and
  unsigned integers only. The `weight_map` names must be exactly the layout's 36734 tensors,
  each once. A shard name is 13 to 127 bytes of `[A-Za-z0-9._-]`, does not start with `.`,
  and ends in `.safetensors`; at most 64 distinct shards. `total_size` must equal 2 bytes
  times the parameter count derived from the config.
- **Shards.** Each file is opened with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW` and must be a regular
  file. The 8-byte little-endian header length is checked against the file size. Each header
  entry needs dtype `BF16` and exactly the shape derived from the config for its slot.
  `data_offsets` must satisfy `begin <= end` and `end - begin == 2 * prod(shape)` without
  overflow. Within a shard the extents must tile `[0, data length)` exactly: no gap, no
  overlap, no trailing bytes. Each tensor must sit in the shard the index names, and every
  tensor the index assigns to a shard must be present in it.
- **Totals.** Parameters are 39785287456, of which 37849399296 are routed expert parameters.
  Both match the issue and the WolframLanguageEvaluator reconciliation recorded with the
  source capture.

The reader reads headers with `pread`. It never maps or reads tensor data and keeps no
duplicate buffers. It does not hash shard payloads: the 15 shard sha256 values are pinned
separately (`shards.lfs-sha256` in the source capture) and checked by a separate step.

### Storage dtype versus compute contract

Every tensor is stored as BF16, including the router `mlp.gate.weight` [256x2048]. The FP32
sigmoid router is a compute contract of the forward path, not a storage dtype. Nothing in
the checkpoint is FP32.

### Canonical slot space

Slot `0` is `model.embed_tokens.weight`, `1` is `model.norm.weight` and `2` is
`lm_head.weight`. Every other tensor is slot `3 + layer * 784 + kind`:

| kind | tensor | layers |
|---|---|---|
| 0, 1 | `input_layernorm`, `post_attention_layernorm` | all |
| 2..5 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | all |
| 6, 7 | `q_norm`, `k_norm` | all |
| 8 | `softmax_offset` (the learned sink) | all |
| 9..11 | dense `gate_proj`, `up_proj`, `down_proj` | 0, 1 |
| 12 | router `mlp.gate` | 2..48 |
| 13..15 | `shared_expert.{gate,up,down}_proj` | 2..48 |
| 16 + 3e + j | `experts.e.{gate,up,down}_proj` (j 0, 1, 2) | 2..48 |

There is no `shared_expert_gate` tensor: the shared expert is ungated.

## Refusals

On refusal nothing is written to stdout. stderr carries `refused N reason` and the exit
status is N.

| Exit | Reason | Meaning |
|---|---|---|
| 2 | usage | not exactly three arguments |
| 3 | io | open, read or stat failed, or a shard is absent |
| 4 | not-regular-file | a symlink (`ELOOP` under `O_NOFOLLOW`), directory or other non-regular file |
| 10 | config-json | config is not one JSON object |
| 11 | config-field | unknown, duplicate, missing or wrong-valued config key |
| 20 | index-size | index over 4194304 bytes |
| 21 | index-syntax | index outside the strict subset |
| 22 | shard-name | shard name outside the allowed form, or more than 64 distinct shards |
| 23 | unknown-tensor | a name outside the layout, in the index or a header |
| 24 | index-duplicate | a name twice in `weight_map` |
| 25 | total-size | `total_size` differs from 2 bytes times the derived parameter count |
| 26 | index-missing-tensor | a layout tensor absent from `weight_map` |
| 30 | shard-short | shard shorter than 8 bytes |
| 31 | header-length | header length under 2, over 983040, or past the end of the file |
| 32 | header-json | header is not valid JSON |
| 33 | header-entry | header root or entry has the wrong shape |
| 34 | header-metadata | `__metadata__` present |
| 35 | dtype | dtype other than `BF16` |
| 36 | shape | rank or dimensions differ from the config-derived shape |
| 37 | data-offsets | `begin > end`, out of range or overflowing, a length that is not 2 x elements, or data truncated |
| 38 | data-tiling | extents overlap, leave a gap, or leave trailing bytes |
| 39 | header-duplicate | a tensor in two headers |
| 40 | shard-mismatch | tensor stored in a different shard than the index says |
| 41 | header-missing-tensor | a tensor the index assigns to a shard is absent from its header |
| 42 | shard-capacity | over 4096 tensors in one shard, or a header that fills the JSON tape |
| 43 | output-capacity | TSV over 8388608 bytes; a defensive guard that no admitted layout reaches (the full output is 4269125 bytes) |

Hugging Face cache snapshots are directories of symlinks, so they are refused with 4. Point
`SHARD_DIR` at real files, for example after `huggingface-cli download --local-dir`.
`__metadata__` is refused because the pinned checkpoint has none.

## Gate: `check.sh`

```sh
OMC=/path/to/omega-mv/seed/out/omc bash om/flint/check.sh
```

`test/gen.py` is the fixture referee. It derives the full tensor layout from `config.json`
alone and writes a sparse 15-shard checkpoint with the published shard boundaries. It also
writes the expected TSV and one mutant per refusal. The gate:

1. builds the reader with the pinned omc, refusing any other compiler hash;
2. requires the synthetic checkpoint to load with a TSV byte-equal to the referee's and the
   exact summary line, and requires usage exit 2 for 0, 1, 2 and 4 arguments;
3. runs 38 refusal fixtures, each required to exit with its own code, print `refused <code>`
   and write nothing to stdout;
4. rebuilds the reader 25 times, each with one guard removed (every `fm_die N` for one code,
   plus `O_NOFOLLOW` dropped from the open flags). Each mutant must stop producing its code on
   that guard's fixture, which proves the refusal comes from the named guard.

Optional arms, which print `NOT_RUN` when unset:

- `FLINT_SRC=dir` is the pinned source capture (`config.json`, the index, `headers/<shard>.header.json`
  and `shards.lfs-sha256`). The arm rebuilds sparse shards from the exact published header bytes
  and sizes, and the TSV sha256 must equal `60ff4dbd…f5`.
- `FLINT_SHARDS=dir` is the 15 downloaded shards, with config and index from `FLINT_SRC`. The
  same TSV sha256 is required.
- The forward-operator arm runs when both are set and the `FLINT_SHARDS` arm wrote its TSV.
  `FLINT_PY` is the referee's interpreter (torch plus the transformers `a672eb5f` checkout,
  e.g. `env PYTHONPATH=/tmp/tf-a672 ~/.venv/bin/python`); default `python3`.

Measured 2026-10-29 on the Spark host CPU (`taskset -c 10-14`), reader sha256
`d9caa2ab4ecf2580114cfdc8cedb30eb93e80f9f1fc0c0da1be1f31a8d381ae8`: `OM-FLINT-MANIFEST PASS (70)`
with both optional arms. On the downloaded checkpoint the reader took 0.27 s wall and
9440 kB max RSS, with 0 major faults. Without the optional arms the result is
`PASS (68)` plus two `NOT_RUN` lines. With the operator arm the full gate is `PASS (89)`.

## Forward operators: `ops.om`

```sh
{ printf 'om2\n'; cat om/vendor/json.om om/flint/ops.om; } | "$OMC" > flint-ops && chmod +x flint-ops
./flint-ops selftest
./flint-ops OP FIXTURE INVENTORY SHARD_DIR      # OP: rmsnorm qkv attn router shared expert moe
```

`test/opref.py` is the referee: it runs the pinned reference's own modules
(`modeling_flint.py` sha256 `f0540832…0c09` and `configuration_flint.py` sha256
`849eee30…4d00`, transformers `a672eb5f`) in f32 on deterministic inputs built from the real
BF16 weights and writes one fixture per operator, with the torch version in header word 6.
`check.sh` requires the fixtures, concatenated in a fixed order, to hash to `FX_PIN`
(`2d838552…7b7a`, torch 2.14.0 on the Spark host CPU); any other torch, reference file or
CPU result fails that check. `ops.om` reads the
fixture, loads the named BF16 tensors from the shards through the inventory TSV, recomputes the
output in f64 and prints `flint-op OP layer L elems N worst_ppm P ties K PASS|FAIL` (exit 0, 1,
or 2 and up for a refusal). An element passes within twice a first-order bound on the
reference's own f32 error; `test/TOLERANCE.md` has the bound per operator, the gamma_n table,
the Wolfram queries and the detection floor (systematic errors below about 2 gamma_n of the
output's last accumulation always pass). The RoPE bound allows FULP = 4 ulps (`ops.om`) per
powf, cosf and sinf call of the reference; `opref.py` measures those calls against float64 and
refuses to write fixtures if any exceeds FULP (2026-10-29: worst 0.848 ulp, cosf).

Fixtures: RMSNorm (layer 2); q/k/v with q/k norm and RoPE on sliding layer 2, and on full
layer 5 without RoPE; attention with sinks over 520 keys, the 513-key window and its edge on
layer 2, full attention on layer 5, through o_proj; the sigmoid top-6 renormalized router on
layer 24, plus a control with the reference top-6 swapped (must FAIL) and a near-tie control
(must PASS with ties above 0); the shared expert, routed expert 99 and the whole MoE block on
layer 24.

Mutants, each required to turn its operator's line to FAIL with exit 1: softmax router
(`om_exp` in place of `om_sig`), shared expert gated by sigmoid(0), sink omitted, RoPE on every
layer, expert index permuted (255 - e), and window 514.

## Not covered (NOT_RUN for this unit)

- bf16-activation parity: the referee runs in f32;
- generation, bf16 activations and long-context serving remain NOT_RUN; complete two-token
  chained layer/intermediate/logit parity is now covered by the optional arm below;
- weight streaming and repacking, with peak allocation and residency measured during load;
- per-expert layout checks before fusing;
- tokenizer and control-token identity, and a load report.

This unit has no GPU arm. The GGUF header importer is the separate unit `gguf.om` below.

## GGUF importer: `gguf.om` (Core #834)

`flint-gguf report GGUF` admits the header of the serving artifact
`~/models/flint-nvfp4-gguf/flint-nvfp4.gguf` (22863084160 bytes, sha256 `97cc8009…6b8bde`) the
way ggml's reader at llama.cpp tag b11804 (commit c7fc5094) does: magic, version 3, every key and
value of all 13 value types (arrays included, nested arrays refused), empty or duplicate keys
refused, `general.alignment` a nonzero power-of-two u32 (default 32), at most 4 dims per tensor
with the element count at most INT64_MAX, a known ggml type whose block divides ne0, unique
names, and a data section where each tensor starts at the padded end of the one before it.
It reads the file from disk only, one bounded pread of at most 16 MiB of header. Stricter than
ggml: names and keys must be printable tokens (names at most 63 bytes, keys at most 255),
bytes past the last padded tensor are refused, and arrays hold at most 999999 elements.

On success stdout is the report and stderr `flint-gguf ok tensors T kvs K bytes F`. The report
format is in the docstring of `test/ggufref.py`, the referee, which writes the same report with
the b11804 `gguf-py` (0.19.0; it checks the sha256 of `gguf_reader.py` and `constants.py`).
The referee's reports are frozen as `test/gguf-synth-report.txt` (the synthetic file from
`test/ggufgen.py`) and `test/gguf-report.txt` (the serving file); `check.sh` pins both and
requires `gguf.om`'s output to equal them byte for byte. The serving report shows 1264
tensors: 485 NVFP4 (type 40, 22192275456 bytes), 48 BF16 (type 30: `token_embd` and the 47
routers), 731 F32 (type 0: norms, sinks and the 485 NVFP4 `.scale` tensors).

| code | reason | cause |
|---|---|---|
| 2 | usage | not `report GGUF`, `map GGUF TSV` or `deq GGUF TSV SHARD_DIR` |
| 3 | io | open, stat, read or close failed |
| 4 | not-regular-file | directory, symlink (O_NOFOLLOW) or device |
| 60 | gguf-truncated | the header ends before a field it declares |
| 61 | gguf-magic | not `GGUF` |
| 62 | gguf-version | version other than 3 |
| 63 | gguf-kv | unknown value type, array of arrays, empty or non-printable key |
| 64 | gguf-kv-duplicate | a key appears twice |
| 65 | gguf-alignment | `general.alignment` not a u32, zero or not a power of two |
| 66 | tensor-shape | 0 or more than 4 dims, element count over INT64_MAX, ne0 not a multiple of the block |
| 67 | ggml-type | type id 43 or above, or a removed id (4, 5, 31..33, 36..38) |
| 68 | tensor-name | empty, longer than 63 bytes, non-printable or duplicate name |
| 69 | tensor-misaligned | offset not a multiple of the alignment |
| 70 | tensor-past-eof | tensor bytes end past the end of the file |
| 71 | tensor-tiling | offset not the padded end of the previous tensor, or bytes after the last |
| 72 | header-capacity | header over 16 MiB, more than 4096 tensors, 1024 keys or 999999 array elements |
| 73 | output-capacity | report over 4 MiB |
| 74 | inventory-tsv | a TSV row not `name shard BF16 R[xC] begin end`, end - begin not 2 · elements, a duplicate name, over 8 MiB or 65536 rows |
| 75 | name-map | a GGUF tensor name with no rule below |
| 76 | tensor-class | GGUF type or dims not those of the tensor's class, or the transposed shape differs from the TSV |
| 77 | coverage | a mapped name absent from the TSV, a TSV row named twice, or a TSV row never named |
| 78 | scale-sibling | an NVFP4 `X.weight` without an F32 `X.scale` of [1] (or [ne2] for `_exps`), or #scales ≠ #NVFP4 |
| 79 | totals | TSV, GGUF or routed element totals differ from the pinned ones |

Gate arm 6 of `check.sh`: the synthetic report equals the referee's; usage, absent file,
directory and symlink refusals; 20 refusal fixtures; 13 guard mutants (each `gg_die N` removed
must stop its fixture producing N) and an O_NOFOLLOW mutant. `FLINT_GGUF=file` compares the
serving report with the frozen one; `FLINT_GGUF_PY=cmd` reruns the referee on both files.
Each is NOT_RUN when unset.

### Map: GGUF tensors against the safetensors inventory

`flint-gguf map GGUF TSV` reads the GGUF header and the TSV that `manifest.om` writes from the
shard headers (the `FLINT_SHARDS` arm's `real.tsv`, sha `TSV_PIN`). Every GGUF tensor whose name
does not end in `.scale` gets one Hugging Face name; a stacked `_exps` tensor gets one name per
routed expert e < ne2. The rules (`blk.N.` becomes `model.layers.N.`):

| GGUF | HF | class |
|---|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` | BF16 matrix |
| `output_norm.weight` | `model.norm.weight` | F32 vector |
| `output.weight` | `lm_head.weight` | NVFP4 matrix |
| `attn_norm` / `ffn_norm` | `input_layernorm` / `post_attention_layernorm` | F32 vector |
| `attn_q` / `attn_k` / `attn_v` / `attn_output` | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | NVFP4 matrix |
| `attn_q_norm` / `attn_k_norm` | `self_attn.q_norm` / `k_norm` | F32 vector |
| `attn_sinks.weight` | `self_attn.softmax_offset` | F32 vector |
| `ffn_gate` / `ffn_up` / `ffn_down` | `mlp.gate_proj` / `up_proj` / `down_proj` | NVFP4 matrix |
| `ffn_gate_inp` | `mlp.gate` | BF16 matrix |
| `ffn_{gate,up,down}_shexp` | `mlp.shared_expert.{gate,up,down}_proj` | NVFP4 matrix |
| `ffn_{gate,up,down}_exps`, expert e | `mlp.experts.e.{gate,up,down}_proj` | NVFP4, 3 dims |

(`.weight` is kept on both sides except `softmax_offset`.) An F32 vector has 1 dim, type 0 and
ne0 = the HF length; a matrix has 2 dims (3 for `_exps`), type 30 (BF16) or 40 (NVFP4), and
HF shape [ne1, ne0]: GGUF ne0 is the HF column count, so the row-major HF matrix and the GGUF
rows are in the same order. Each name must hit exactly one TSV row, every TSV row must be hit,
each NVFP4 tensor needs its `.scale` sibling, and the totals are pinned: 39785287456 elements
(Wolfram `216864+332136448+39452934144` → 39785287456, the F32, BF16 and NVFP4 sums) and
37849399296 routed (Wolfram `141*256*512*2048` → 37849399296). The serving file gives
`map units 36734 tensors 779 scales 485 elements 39785287456 routed 37849399296`: 638 plain
tensors plus 141 `_exps` tensors × 256 experts = 36734 units, the TSV's row count.

### Deq: sampled NVFP4 blocks against the BF16 shards

`flint-gguf deq GGUF TSV SHARD_DIR` runs map, then compares GGUF data with the BF16 bytes of
each unit's TSV row (the shard file `SHARD_DIR/<shard>`, offset 8 + header length + begin):

- F32 vectors: every value; the GGUF f32 must equal the BF16 value widened, bit for bit.
- BF16 matrices: 8 sampled 64-value chunks per tensor, byte for byte.
- NVFP4 units (tensors, and each expert slice): 8 sampled 36-byte blocks per unit, each of its
  64 values under the frozen bound below. The block of row r, block k of expert e is at
  `offset + e · (ne0 · ne1 / 64 · 36) + (r · ne0/64 + k) · 36`; the scale is the F32 at
  `.scale` offset + 4e.

Sample k of a unit: k = 0 is row 0 block 0, k = 1 the last row's last block, k = 2..7 come from
the 64-bit LCG x ← x · 6364136223846793005 + 1442695040888963407 (seed 834 once, continued over
units in GGUF tensor order, then expert order) with row = (x >> 33) mod rows and block =
(x >> 11) mod (ne0/64). The output line is `deq units U blocks B elements N violations V
strict S worst_ppm W exact X mismatches M PASS|FAIL`: strict counts values over the bound
without its BF16 term, worst_ppm is the largest |w − ŵ| / bound × 10^6, exact and mismatches
count the F32 and BF16 values compared bit for bit. A violation also prints the first one
(tensor, expert, row, column, |w − ŵ| / D and bound / D in ppm). FAIL exits 1.

Gate arm 6, when `FLINT_GGUF` and the TSV are present: the map line; three TSV mutants on
`model.layers.2.mlp.experts.0.down_proj.weight` (row dropped → 77, shape 512x2048 → 76, dtype
F16 → 74); the synthetic GGUF → 75; source mutants expert-dup (`gg_hu or e 1` → 77) and totals
(→ 79). With `FLINT_SHARDS` too: deq must PASS, and three source mutants must FAIL it with at
least ten times the unmutated run's violations (and at least 1000), so a failing unmutated run
cannot pass them by default: expert-swap (`gg_hu sub 255 e`, which still maps),
nibble-hi-for-lo (the low nibble decoded from the high one) and scale-order (sub-scales of a
block read in reverse). Not covered: the unsampled NVFP4 blocks (291520 of 616452096 are
sampled; Wolfram `N[291520/(39452934144/64)*100, 4]` → 0.04729 %), serving or GPU evidence.

### Observed on the serving file: deq FAILS against `flint-2a247be0`

Run on 2026-10-30 (host, cores 0-4, 62 s, 24 MB RSS):

    deq units 36734 blocks 291520 elements 18657280 violations 7782 strict 19841 worst_ppm 1394455 exact 241440 mismatches 48 FAIL
    first_violation blk.0.attn_q.weight expert 0 row 2566 col 1841 err_over_d_ppm 1022222 bound_over_d_ppm 1019444

7782 of 18657280 sampled NVFP4 values exceed the frozen bound (Wolfram → 0.04171 %), the worst by
1.394 × the bound. The bound stays as committed. The cause is the reference, not the decoder:
48 of the 241440 values that need no quantization (F32 norms and sinks, BF16 router and
embedding chunks) differ from the shards bit for bit, which no NVFP4 decode can cause. A
throwaway numpy probe over whole tensors (not part of the gate) found 7983313 of 307494912
`token_embd` values different (Wolfram → 2.596 %), differing elements in `output_norm` (7 of
2048), and in `attn_q_norm`, `attn_k_norm` and `attn_sinks` of layers 19, 20, 28, 32, 34, 35,
38 and 48 (one to four each, e.g. 0.00098419 against 0.00086594 in layer 28's k norm); all 47
routers equal. On `blk.0.attn_q.weight` the violators all sit at the 4/6 rounding midpoint
(|w/D| from 4.74 to 5.19, decoded as 4 or 6); requantizing the shard weights to the nearest
E2M1 value agrees with 98.17 % of the file's codes; the block maximum over 6 · D has median
1.0007. So the GGUF was quantized from weights close to, but not equal to, the released
checkpoint (`RecursiveSI/flint` has one revision, 2a247be0; the GGUF is
`RecursiveSI/flint-nvfp4-gguf@6ec43e19`). The tie from the serving file to the checked weights
is therefore exact for names, shapes, layout and totals, and approximate (not within the frozen
bound) for values.

### NVFP4 dequantization tolerance (derived from the format before any full comparison)

ggml b11804 (`dequantize_row_nvfp4` in `ggml/src/ggml-quants.c`, `ggml_ue4m3_to_fp32` in
`ggml-impl.h`) decodes one 36-byte block of 64 elements as four 16-element sub-blocks:
`y[16s + j] = E2M1(qs[8s + j] & 15) · ue(d[s])` and `y[16s + j + 8] = E2M1(qs[8s + j] >> 4) · ue(d[s])`
for j < 8. Here E2M1 is the value set ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. ue(x) is 0 for x = 0 or 127. Otherwise, with
e = (x >> 3) & 15 and m = x & 7, it is m · 2^-9 when e = 0 and (1 + m/8) · 2^(e-7) when e > 0. The
llama.cpp graph multiplies by the tensor's F32 `.scale` (one value, or one per expert for `_exps`).
So the dequantized weight is ŵ = q · D with D = ue(d[s]) · scale.

The bound for a reference BF16 weight w is

    |w - ŵ| <= 1 · D + max(0, |w| - 6 · D) + 2^-8 · |w|

It has three terms:

- **1 · D:** a nearest-value quantizer's worst error on the E2M1 grid is half the largest grid gap.
  The largest gap is the top binade, 4 to 6.
  Wolfram: `g={0,1/2,1,3/2,2,3,4,6}; {Max[Differences[g]]/2, Position[Differences[g], Max[Differences[g]]]}` → `{1, {{7}}}`.
- **max(0, |w| - 6 · D):** clipping. When the E4M3 sub-scale was rounded down, the block's largest
  value lies beyond 6 · D.
- **2^-8 · |w|:** the rounding of the reference to BF16 (8-bit significand, unit roundoff 2^-8).
  Wolfram: `2^(1-8)/2` → `1/256`.

The comparison is in f64. q · ue · scale is exact there (significand bits: 2 + 4 + 24 = 30, at most 53).

The pass rule is zero violations over the sampled blocks. The bound was fixed and committed before
the full comparison. An earlier 30-block probe (max 1.031 · D, without the 2^-8 term) was
exploratory, and the bound does not use it. If real tensors exceed the bound, the gate reports
FAIL with the numbers; the bound is not loosened.

## Complete bounded host forward

`FLINT_FORWARD=1` adds a full 49-layer chained computation to the same check command.
It starts from two token IDs, loads real BF16 embedding rows and all selected expert weights,
and compares five intermediate tensors per layer, final normalization and every vocabulary logit.
The reference tensors are comparison targets, never teacher-forced inputs.
The full gate prints `OM-FLINT-MANIFEST PASS (92)` with real checkpoint inputs; wrong-token
and skipped-final-normalization source mutants must fail the comparison.

See [PROFILE.md](PROFILE.md) for the frozen weight/runtime/tokenizer/template identities
and the architecture audit. In particular, this checkpoint has **no expert bias**, unlike
the requested selection-only bias description. [test/TOLERANCE.md](test/TOLERANCE.md)
declares the candidate integration budget separately from the tighter per-operator bounds.
Native f64 computation with BF16 storage is compared to reference f32 computation.
This is host-forward evidence only: no serving API, task completion, quantized model,
KV-cache, long-context batching or GPU/backend qualification is claimed.
