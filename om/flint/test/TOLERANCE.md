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
| RoPE (`qkv-2` sliding layer; `qkv-5` full layer has none) | cos/sin of `theta = pos inv_freq`: `theta gamma_4 + gamma_2`; rotation `gamma_2` | 4, 2 |
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

Router near-tie rule: with sigmoid scores ranked by the .om side, a token whose rank-6/rank-7
margin is under `2 x (2 (E_6 + E_7))` is reported as a tie. On a tie token, a reference index
missing from the .om top-6 is accepted only if it equals the .om rank-7 index; any other missing
index fails. The MoE operator skips tie tokens and counts them in `ties`.

## Wolfram queries

`WolframLanguageEvaluator: u = 2^-24; g[n_] := n u/(1 - n u); {N[u, 20], Table[{n, N[g[n], 12]}, {n, {1, 2, 4, 7, 8, 130, 134, 512, 520, 524, 1024, 2048, 2054, 4096}}], N[MaxValue[D[x LogisticSigmoid[x], x], x], 20]}`

| n | gamma_n |
|---|---|
| 1 | 5.96046483281e-8 |
| 2 | 1.19209303762e-7 |
| 4 | 2.38418635945e-7 |
| 7 | 4.17232687511e-7 |
| 8 | 4.76837385577e-7 |
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
at relative 2e-15, and the RoPE angle `519 x 10000^(-2/128)` (pos 519, pair 1) at 1e-12. omc
float literals hold at most 2^63 - 1 in the digit string (error 8 above that), so five of the
values are written with 18 significant digits in `ops.om`. The softmax scale is
`1/sqrt(128)` and `inv_freq_i = exp(-i ln(10000) / 64)`; omc has no exp, sin or cos, so
`ops.om` implements them (Cody-Waite reduction, Taylor polynomials).

## Measured (Spark host CPU, `taskset -c 10-14`, 2026-10-29)

```
flint-op selftest layer 0 elems 13 worst_ppm 51902 ties 0 PASS
flint-op rmsnorm layer 2 elems 32768 worst_ppm 690 ties 0 PASS
flint-op qkv layer 2 elems 40960 worst_ppm 5212 ties 0 PASS
flint-op qkv layer 5 elems 40960 worst_ppm 886 ties 0 PASS
flint-op attn layer 2 elems 49152 worst_ppm 1876 ties 0 PASS
flint-op attn layer 5 elems 49152 worst_ppm 5017 ties 0 PASS
flint-op router layer 24 elems 16768 worst_ppm 1443 ties 16 PASS
flint-op shared layer 24 elems 8192 worst_ppm 16 ties 0 PASS
flint-op expert layer 24 elems 8192 worst_ppm 17 ties 0 PASS
flint-op moe layer 24 elems 6144 worst_ppm 6 ties 1 PASS
```

16 of the 64 router tokens fall inside the tie band: `gamma_2048` is a worst-case bound, and
the band is wider than the observed error. The tie tokens still have their weights compared
whenever both top-6 sets agree. The reference runs in f32; bf16-activation parity is not
measured here.
