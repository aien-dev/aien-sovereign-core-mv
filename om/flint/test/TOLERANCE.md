# Forward-operator tolerances (Core #802)

`ops.om` recomputes each operator in f64 from the real BF16 weights and carries, per output
element, a first-order bound E on the error of the reference's own f32 computation. An element
passes when `|om - ref| <= 2 E`; a NaN fails. `worst_ppm` is `max |om - ref| / (2 E)` in
millionths, so every PASS line has `worst_ppm < 1000000`.

Unit roundoff for f32 accumulation is `u = 2^-24`, and `gamma_n = n u / (1 - n u)`. A dot product
of length K with inputs that carry their own bounds E_x gets
`E = sum |w| E_x + gamma_K sum |w x|`.

## n per operator

| Operator (fixture) | Bound terms | n |
|---|---|---|
| RMSNorm (`rmsnorm`, layer 2 `input_layernorm`, T 16) | `gamma_(n+6) |y|` plus propagated input error | 2054 (n = 2048) |
| q/k/v projection (`qkv-2`, `qkv-5`, S 8) | dot over hidden size | 2048 |
| q_norm, k_norm (inside `qkv`) | RMSNorm over head_dim | 134 (n = 128) |
| RoPE (`qkv-2` sliding layer; `qkv-5` full layer has none) | cos/sin of `theta = pos inv_freq` under a libm budget of FULP = 4 ulps per call: `theta gamma_(2 FULP + 2) + gamma_(2 FULP)` = `theta gamma_10 + gamma_8`; rotation `gamma_2` | 10, 8, 2 |
| attention scores (`attn-2`, `attn-5`, T 520 keys, S 8 queries) | `gamma_130 scale sum |q k|` (128-term dot, scale) | 130 |
| softmax with sink | per score `E_s + E_max + u |s - max| + gamma_2`; normalizer weighted by `e_k`; `gamma_(T+4)` | 524 (T = 520) |
| P·V | `sum p_j |v| (rho_j + gamma_T)` | 520 |
| o_proj | dot over 32 x 128 | 4096 |
| router logits (`router`, layer 24, T 64) | dot over hidden size | 2048 |
| sigmoid score | `sigma (1 - sigma) E_l + gamma_4 sigma` | 4 |
| top-6 renormalized weight | `(E_k + w sum E) / S + gamma_7 w` | 7 |
| gate/up (`shared`, `expert` 99, layer 24, T 4) | dot over hidden size | 2048 |
| silu and gate*up | `1.1 E_g + gamma_4 |s|` (max silu' below); `gamma_1 |h|` | 4, 1 |
| down projection, shared expert | dot over intermediate 1024 | 1024 |
| down projection, routed expert | dot over intermediate 512 | 512 |
| MoE sum (`moe`, layer 24, T 4) | `sum (|w| E_y + |y| E_w) + E_shared + gamma_8 (sum |w y| + |shared|)` | 8 |

Router near-tie rule: with sigmoid scores s ranked by the .om side (ranks 1..7), a token is a tie
when some .om top-6 expert k has `s_k - s_7 < 2 (E_k + E_7)`. On a tie token, a reference index
missing from the .om top-6 is accepted only if it equals the .om rank-7 index; any other missing
index fails. The MoE operator skips tie tokens and counts them in `ties`.

Why `2 (E_k + E_7)` and no more. The logits are compared element by element, so a token that
passes has `|l_om - l_ref| <= 2 E_l` for all 256 experts. The score bound
`E_s = sigma (1 - sigma) E_l + gamma_4 sigma` then gives, to first order, `|s_om - s_ref| <= 2 E_s`
(the doubled logit term through sigma' plus the reference's own sigmoid rounding). The reference
can put the .om rank-7 expert in its top-6 only by ranking it above some .om top-6 expert k:
`s_om,k - 2 E_k <= s_ref,k < s_ref,7 <= s_om,7 + 2 E_7`, that is `s_k - s_7 < 2 (E_k + E_7)`.
Outside that band such a reference top-6 is a real mismatch, so the band needs no further factor.
The earlier rule (until 6a95f6f8) tested only k = rank 6 against `2 x 2 (E_6 + E_7)`; that extra
factor 2 had no derivation, and it hid tokens from the MoE comparison. A reference rank-8 or lower
expert in its top-6 is still a FAIL, never accepted: it can cause a false FAIL, not a false PASS.

Measured on the router fixture (64 tokens): the old band flagged 16, the derived band flags 10,
and the boundary pair alone (k = rank 6 with `2 (E_6 + E_7)`) also flags 10, so no token's tie
comes from a non-boundary k here. The MoE fixture now compares all 4 tokens (elems 8192, ties 0;
before: 3 tokens, elems 6144, ties 1). router-tie keeps PASS with ties 1, router-swap keeps FAIL
at the baseline worst_ppm 1443 (index check, not a numeric change), and with the accept rule
forced to reject (`if true` in `rt_cmp`) router-tie FAILs while router still PASSes, so the accept
path is exercised by router-tie and by none of the 10 tie tokens of the router fixture.

## RoPE libm budget

The reference computes `inv_freq = 1.0 / (theta ** (arange(0, 128, 2) / 128))` and then
`cos(pos inv_freq)`, `sin(pos inv_freq)` in f32 (`modeling_flint.py` `FlintRotaryEmbedding`).
The exponent is exact, the division and the product each round once, and powf, cosf and sinf
each err by at most FULP ulps of their result, where 1 ulp is at most `2u` relative. To first
order the angle then carries relative error `gamma_(2 FULP + 2)`, an absolute `theta gamma_(2 FULP + 2)`
that moves cos and sin by at most as much, and the cos/sin calls add `gamma_(2 FULP)` absolute. `ops.om` sets
`const FULP ulong 4`.

Why 4: the glibc 2.39 manual ("Known Maximum Errors in Math Functions",
https://sourceware.org/glibc/manual/2.39/html_node/Errors-in-Math-Functions.html) lists 1 ulp for
AArch64 `powf`, `cosf` and `sinf`; SLEEF, which torch's vectorized CPU kernels can use, documents
1.0 ulp for its `_u10` functions and 3.5 ulps for `_u35` (https://sleef.org/). FULP 4 holds for
any of these. The earlier bound `theta gamma_4 + gamma_2` (until e6c022b6) assumed 1 ulp per call
and left no room for a `_u35` path.

The budget is also measured, not only assumed. `test/opref.py` reads FULP from `ops.om`, compares
the reference module's own `theta ** expo`, `cos` and `sin` over positions 0..519 with float64, and
exits before writing the q/k/v and attention fixtures when any call exceeds FULP ulps. On the Spark
host CPU with torch 2.14.0 it prints `rope ulps pow 0.492 cos 0.848 sin 0.708 of 4`, so the worst
call uses 0.212 of the budget (`WolframLanguageEvaluator: N[{0.492/4, 0.848/4, 0.708/4}, 6] -> {0.123, 0.212, 0.177}`).
The selftest pins the bound at pos 519, pair 1:
`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); th = 519 10000^(-1/64); NumberForm[N[{th, th g[10] + g[8], th g[4] + g[2]}, 25], 19] -> {449.4354838238739179, 0.0002683614206850094353, 0.0001072730043023284837}`,
2.50 times the 1-ulp value (`WolframLanguageEvaluator: N[2.683614206850094353*^-4/1.072730043023284837*^-4, 6] -> 2.50167`).
With FULP 4, `qkv-2` worst_ppm went from 5212 to 4826 (`WolframLanguageEvaluator: N[4826/5212, 6] -> 0.925940`), and the
rope-global mutant on `qkv-5` still FAILs at worst_ppm 481023464. Controls on edited copies:
FULP 1 restores `qkv-2` worst_ppm 5212 and turns the selftest to FAIL (`FAIL (1 of 89)`); FULP 0
makes `opref.py` exit with `RoPE libm ulps pow 0.492 cos 0.848 sin 0.708 exceed FULP 0`
(`FAIL (1 of 71)`).

## Detection floor

The bound is a worst case, so it also limits what the check can see. Take an output y whose last
accumulation has length n. That term alone gives `E >= gamma_n |y|` (for a dot product,
`gamma_n sum |w x| >= gamma_n |y|`; earlier stages only add to E). If the clean run's worst element
uses a fraction `w = worst_ppm / 10^6` of the tolerance `2 E`, then an .om error that scales y by
`1 + delta` stays inside the tolerance on every element whenever `delta <= (1 - w) 2 gamma_n`.
Systematic relative errors below that floor always pass. Above it, an error is caught only where
there is little cancellation (`sum |w x|` close to `|y|`). Errors in an internal parameter reach the
outputs scaled by the operator's sensitivity, which can be well below 1.

`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); rows = {{"rmsnorm y", 2054, 690}, {"qkv-2 v", 2048, 4826}, {"qkv-2 q,k", 134, 4826}, {"qkv-5 v", 2048, 886}, {"qkv-5 q,k", 134, 886}, {"attn-2 o", 520, 1876}, {"attn-2 y", 4096, 1876}, {"attn-5 o", 520, 5017}, {"attn-5 y", 4096, 5017}, {"router logits", 2048, 1443}, {"router w", 7, 1443}, {"shared y", 1024, 16}, {"expert y", 512, 17}, {"moe y", 512, 6}}; Map[{#[[1]], #[[2]], #[[3]], N[(1 - #[[3]]/10^6) 2 g[#[[2]]], 6]} &, rows]`

| Fixture output | n of last accumulation | clean worst_ppm | floor `(1 - w) 2 gamma_n` |
|---|---|---|---|
| `rmsnorm` y | 2054 | 690 | 2.44717e-4 |
| `qkv-2` v | 2048 | 4826 | 2.42992e-4 |
| `qkv-2` q, k (after q/k norm) | 134 | 4826 | 1.58971e-5 |
| `qkv-5` v | 2048 | 886 | 2.43954e-4 |
| `qkv-5` q, k (after q/k norm) | 134 | 886 | 1.59600e-5 |
| `attn-2` o (P·V) | 520 | 1876 | 6.18745e-5 |
| `attn-2` y (o_proj) | 4096 | 1876 | 4.87484e-4 |
| `attn-5` o (P·V) | 520 | 5017 | 6.16797e-5 |
| `attn-5` y (o_proj) | 4096 | 5017 | 4.85950e-4 |
| `router` logits | 2048 | 1443 | 2.43818e-4 |
| `router` top-6 weights | 7 | 1443 | 8.33261e-7 |
| `shared` y (down, 1024) | 1024 | 16 | 1.22076e-4 |
| `expert` y (down, 512) | 512 | 17 | 6.10360e-5 |
| `moe` y (`E >= gamma_512 |y|` through the expert and shared terms) | 512 | 6 | 6.10367e-5 |

These are lower bounds: the propagated terms of earlier stages raise each E, so the real floor of
a chained output (q/k after the norm, o, y, shared, expert, moe) is higher. A uniform scale error
in the q/k projections is invisible after q/k norm at any size, because RMSNorm is
scale-invariant. Review evidence on 6a95f6f8: a softmax scale of 0.0884 in place of `1/sqrt(128)`
(relative `WolframLanguageEvaluator: N[(0.0884 - 1/Sqrt[128])/(1/Sqrt[128]), 6] -> 0.000131831`)
passes `attn-5` at worst_ppm 942555, and an RMSNorm epsilon of 1e-5 in place of 1e-6 passes
`qkv-2` (worst_ppm 268470; q/k norm inputs have a large mean square) while `rmsnorm` (raw
embedding rows) catches it. The six forward mutants in `check.sh` are structural and land 1.8x
to 1500x above the tolerance (`WolframLanguageEvaluator: N[{1788241, 11564696, 20605613, 69071141, 481023464, 1497684841}/10^6, 6] -> {1.78824, 11.5647, 20.6056, 69.0711, 481.023, 1497.68}`).
This check therefore proves the operators' structure and their
agreement to about the f32 worst-case bound (relative 2.4e-4 at n = 2048,
`WolframLanguageEvaluator: N[2 2048 2^-24/(1 - 2048 2^-24), 6] -> 0.000244170`); it does not
detect systematic errors smaller than that.

## Wolfram queries

`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); {N[u, 20], Table[{n, N[g[n], 12]}, {n, {1, 2, 4, 7, 8, 130, 134, 512, 520, 524, 1024, 2048, 2054, 4096}}], N[MaxValue[D[x LogisticSigmoid[x], x], x], 20]}`

| n | gamma_n |
|---|---|
| 1 | 5.96046483281e-8 |
| 2 | 1.19209303762e-7 |
| 4 | 2.38418635945e-7 |
| 7 | 4.17232687511e-7 |
| 8 | 4.76837385577e-7 |
| 10 | 5.96046803025e-7 (`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); N[g[10], 12]`) |
| 130 | 7.74866386213e-6 |
| 134 | 7.98708619294e-6 |
| 512 | 3.05185094760e-5 |
| 520 | 3.09953759668e-5 |
| 524 | 3.12338093827e-5 |
| 1024 | 6.10388817677e-5 |
| 2048 | 1.22085215480e-4 |
| 2054 | 1.22442930804e-4 |
| 4096 | 2.44200244200e-4 |

`u -> 5.9604644775390625e-8`; `max silu' -> 1.0998393201288669170` (the bound uses 1.1).
The kernel printed `Infinity::indet` messages while evaluating the derivative at the infinite
ends of the domain; the returned maximum is finite.

`WolframLanguageEvaluator: N[{Exp[1], Exp[-81/4], Exp[65/8], Exp[-175/2], Sin[1], Cos[1], Sin[2075/4], Cos[2075/4], Sin[1/1000], Cos[3], LogisticSigmoid[3/2], Cos[519 10000^(-2/128)], Sin[519 10000^(-2/128)], 1/Sqrt[128], Log[10000], 2^-24}, 19]`

```
{2.718281828459045235, 1.605228055185611609e-9, 3377.867931673534884, 9.982350930569247556e-39,
 0.8414709848078965067, 0.5403023058681397174, -0.3776084404242524912, -0.9259653696118250388,
 0.0009999998333333416667, -0.9899924966004454573, 0.8175744761936436596, -0.9824296005444783408,
 -0.1866335446108676431, 0.08838834764831844055, 9.210340371976182736, 5.9604644775390625e-8}
```

The selftest (`flint-ops selftest`) checks `om_exp`, `om_sc` and `om_sig` against these values
at relative 2e-15, and the RoPE angle `519 x 10000^(-2/128)` (pos 519, pair 1) and its error
bound (RoPE libm budget above) at 1e-12. omc
float literals hold at most 2^63 - 1 in the digit string (error 8 above that), so five of the
values are written with 18 significant digits in `ops.om`. The softmax scale is
`1/sqrt(128)` and `inv_freq_i = exp(-i ln(10000) / 64)`; omc has no exp, sin or cos, so
`ops.om` implements them (Cody-Waite reduction, Taylor polynomials).

## Measured (Spark host CPU, `taskset -c 10-14`, 2026-10-29)

```
flint-op selftest layer 0 elems 14 worst_ppm 51902 ties 0 PASS
flint-op rmsnorm layer 2 elems 32768 worst_ppm 690 ties 0 PASS
flint-op qkv layer 2 elems 40960 worst_ppm 4826 ties 0 PASS
flint-op qkv layer 5 elems 40960 worst_ppm 886 ties 0 PASS
flint-op attn layer 2 elems 49152 worst_ppm 1876 ties 0 PASS
flint-op attn layer 5 elems 49152 worst_ppm 5017 ties 0 PASS
flint-op router layer 24 elems 16768 worst_ppm 1443 ties 10 PASS
flint-op shared layer 24 elems 8192 worst_ppm 16 ties 0 PASS
flint-op expert layer 24 elems 8192 worst_ppm 17 ties 0 PASS
flint-op moe layer 24 elems 8192 worst_ppm 6 ties 0 PASS
```

10 of the 64 router tokens fall inside the tie band: `gamma_2048` is a worst-case bound, and
the band is wider than the observed error. The tie tokens still have their weights compared
whenever both top-6 sets agree. The reference runs in f32; bf16-activation parity is not
measured here.

## Full chained forward (candidate numerical profile, 2026-10-29)

`ops.om forward` consumes only token IDs as inputs. It loads real embedding rows, executes
all 49 decoder layers using its own prior outputs, applies final RMSNorm and streams every
vocabulary row of the untied LM head. No teacher forcing or reference router-index adoption.
Each layer compares input norm, attention output, post-attention norm, MLP output and residual
output against the independent pinned reference. Embeddings and final norm are also compared.
The admitted fixture has two tokens, IDs 1001 and 8920. This is a bounded host verifier,
not a serving engine, KV-cache implementation, long-context benchmark or device qualification.
The existing attention operator fixture separately exercises the 513-token window boundary.

Before the first execution, the integration criterion was fixed as
`|native-reference| <= 8 gamma_2048 (1+|reference|)`. This is an explicit integration
budget, **not** a derived worst-case error theorem for 49 layers. The local operator
tests above retain their tighter propagated bounds. `fw_cmp` passes half this budget
to `cmp`, which already compares against twice the supplied bound.
BF16 checkpoint storage is unchanged; native computation is f64, reference computation
is f32, so this result does not establish BF16-activation, quantized or GPU parity.
There is no numerical tuning or relaxed threshold after observing the fixture.

WolframLanguageEvaluator:
`{49-2,49-Length[{0,1,5,9,13,17,21,25,29,33,37,41,45,48}], N[2^(-24)*2048/(1-2^(-24)*2048),18], 2*2048*150144, 3*512*2048*2, 6*3*512*2048*2, 49*2*2048*4*5+2*150144*4}`
returned
`{47,35,0.000122085215480405323,615038976,6291456,37748736,5218304}`.
The byte counts are modeled layout sizes, not measured memory usage.
The chained fixture is pinned by SHA256 in `check.sh`; wrong-token and skipped-final-normalization
source mutants must compile and return comparison exit 1, not a trap or setup refusal.

## Greedy generation parity (Core #834 item 5), frozen before the first comparison

`ops.om generate` starts from the token IDs of a chat-template-rendered prompt and generates N
tokens greedily (argmax of its own f64 logits, lowest index on an exact tie), feeding its own
tokens back through the KV cache, one host forward per position. `test/genref.py` is the referee:
the pinned transformers modules in f32 on CPU, the same prompt IDs, greedy argmax with a
`DynamicCache`, one position per step after the prompt. Per step k = 1..N the fixture records the
reference top-1 ID and logit `l1` and the top-2 ID and logit `l2` (f32).

Per-logit budget: `tau(x) = 8 gamma_2048 (1 + |x|)`, the integration budget of the chained forward
above (`fw_cmpat`), unchanged.
`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); N[8 g[2048], 18] -> 0.000976681723843242583`.

Ambiguity is a property of the reference alone, declared before the .om run: step k is
**ambiguous** when `l1 - l2 <= tau(l1) + tau(l2)`, i.e. when an .om logit vector that meets the
budget on both entries could rank the reference top-2 first. `genref.py` prints the ambiguous
steps when it writes the fixture, and the gate prints them again before the .om run; the rule is
not changed after a run.

Per step, in order, while every earlier step matched (so the .om prefix equals the reference prefix):

1. Logits: the .om logits at the reference top-1 and top-2 IDs must each be within `tau` of `l1`
   and `l2`; otherwise FAIL (logit).
2. Token, non-ambiguous step: the .om argmax must equal the reference top-1; otherwise FAIL (token).
3. Token, ambiguous step: the .om argmax equal to the reference top-1 continues; equal to the
   reference top-2 ends the comparison at step k without FAIL (the sequences no longer share a
   prefix, so later fixture steps do not apply), and the receipt reports k - 1 matched steps, which
   is then not N-token parity; any other ID is FAIL (token).

The verdict line is `flint-op generate layer 49 elems S worst_ppm P ties A PASS|FAIL`, where S
counts compared logits (two per compared step), P is the worst `|om - ref| / tau` over the compared top-1/top-2 logits in
millionths, and A counts the declared ambiguous steps of the fixture. Every step also prints its
.om top-1/top-2 IDs and logits next to the reference values, so a divergence comes with its logits.

The KV cache holds FKVP positions per layer; a request with prompt + N - 1 > FKVP positions is
refused (`refused 7 kv-cache-bound`) at the first position past the bound, never overrun. This is
a host check (f64 compute, BF16 storage, f32 referee); it is not bf16-activation, GPU or serving
evidence.
