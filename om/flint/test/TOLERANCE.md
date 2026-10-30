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

## bf16-activation forward arm (Core #834 item 6, declared 2026-10-30 before the first comparison)

The f32 arm above upcasts every module to f32. The checkpoint ships as bfloat16
(`config.json` `torch_dtype: bfloat16`), so a second referee arm, `fwdref.py ... bf16`
(fixture op 9), runs the same two-token, 49-layer forward with every parameter in bfloat16
except the model's `_keep_in_fp32_modules_strict` (`gate.weight`, `sinks`), and bfloat16
activations. The pinned modules then round an f32 intermediate to bfloat16 (round to nearest,
ties to even) at these points, and nowhere else:

| Site (pinned `modeling_flint.py`) | Rounded to bf16 |
|---|---|
| `FlintRMSNorm` (input, post-attention, q_norm, k_norm, final) | output `(weight * x rsqrt(..)).to(bf16)`; mean square and rsqrt stay f32 |
| every `nn.Linear` (q/k/v/o_proj, dense MLP, shared and routed experts, lm_head) | output, f32 accumulation |
| RoPE (`FlintRotaryEmbedding`, `apply_rotary_pos_emb`) | cos, sin; `q cos`; `rotate_half(q) sin`; their sum (k likewise) |
| eager attention | softmax probabilities (`.to(value.dtype)`) and the `P V` output; scores, sink and softmax stay f32 |
| MLP / experts | `silu(gate)`, `silu(gate) up` |
| `FlintExperts` | `y_e w_e` (bf16 times f32 weight is f32, then `.to(bf16)`); each `index_add_` into the bf16 accumulator, experts in ascending index order |
| `FlintSparseMoeBlock` | routed sum plus shared expert |
| `FlintDecoderLayer` | both residual adds |

Router logits, sigmoid, top-6 and renormalized weights stay f32 (`hs.float()`, f32 weight).
Embedding rows are bf16 on disk and exact. Every value the bf16 arm writes is a bfloat16.
These semantics were probed on the pinned torch 2.14.0+cu130 CPU build before writing the arm:
bf16 `F.linear`, `matmul` and `silu` equal `bf16(f32 or f64 result)` element for element on
random inputs, `bf16 * f32` promotes to f32, and `torch.cat` of f32 and bf16 is f32.

**What ops.om compares (option a).** `ops.om fwbf16` runs the same f64 forward with a bf16
mode: `bfr` rounds to f32 and then to bfloat16 (nearest, ties to even, on the f32 bit
pattern, as torch does) at exactly the sites in the table, and sums routed experts in ascending
index order. Option b, the plain f64 forward against the bf16 reference, cannot see the rounding
mode at all: a native forward that rounded wrongly, or not at all, would be judged by the same
budget as one that rounds correctly. Under option a both sides round at the same sites, so most
compared elements should be bit-identical and the remaining differences are flips of one bf16
ulp where the f32 (reference) and f64 (native) pre-rounding values straddle a rounding
boundary. The budget below then bounds how those flips propagate.

**Budget (declared before measurement).** Per compared element
`|native - reference| <= 8 (gamma_2048 + 2 u_bf16) (1 + |reference|)`, `u_bf16 = 2^-8`.
Derivation: the f32 arm budget is 8 per-operator units of `gamma_2048` (the largest f32
accumulation in the forward) and the measured f32 chained forward used 694277 ppm of it, so a
factor of 8 units held 49 chained layers with 31 percent headroom. In the bf16 arm each
operator output can additionally differ by one bf16 ulp, at most `2 u_bf16` relative, after the
same pre-rounding difference; the per-operator unit becomes `gamma_2048 + 2 u_bf16` and the
factor 8 is kept. Like the f32 arm, this is an explicit integration budget, **not** a derived
worst-case theorem; it is not tuned after observing the fixture. `fw_cmp` passes half of it to
`cmp`, which compares against twice the supplied bound.

WolframLanguageEvaluator:
`u32 = 2^-24; u16 = 2^-8; g = 2048 u32/(1 - 2048 u32); b32 = 8 g; d32 = 694277/10^6 b32; b16 = 8 (g + 2 u16); {N[g, 18], N[b32, 18], N[d32, 12], N[2 u16, 18], N[b16, 18], N[b16/b32, 12], N[(g + 2 u16)/g, 12], N[b16/2, 18], N[d32/b16, 12]}`
returned
`{0.000122085215480405323, 0.000976681723843242583, 0.000678087657185, 0.0078125, 0.0634766817238432426, 64.9921875, 64.9921875, 0.0317383408619216213, 0.0106824685659}`.
So the bf16 budget is 0.0635 (1 + |ref|), 65.0 times the f32 arm budget; the f32 arm's measured
worst difference would use 1.07 percent of it.

**Declared checks and predictions.**
- `fwbf16` on the bf16 fixture: predicted PASS. A router top-6 change caused by a flipped
  bf16 input would exceed the budget and is reported as FAIL, not as a tie.
- Fixture admission: `fwbf16` refuses any fixture word that is not a bfloat16 (low 16 bits
  nonzero) and an op-8 (f32) fixture; `forward` refuses an op-9 fixture.
- Mutant `bf16-truncate` (`bfr` truncates toward zero instead of rounding to nearest even):
  predicted FAIL, because truncation biases every rounding toward zero and the residual stream
  is rounded 98 times without renormalization.
- Mutant `bf16-skip-final-norm`: predicted FAIL.
- Control `bf16-off` (`bfr` is the identity, the plain f64 forward against the bf16 fixture,
  i.e. option b) is measured and recorded whichever way it lands; it is not a gate entry,
  because this budget was not derived to separate it.

**v1 measured: FAIL at layer 1 (2026-10-30, fixture `ac10d325...`, host CPU `taskset -c 0-4`).**
The v1 budget above is kept unedited as declared in `9fb7d7ab`. `ops fwbf16` printed
`flint-op fwbf16 layer 0 elems 24576 worst_ppm 111729 ties 0 PASS` and then
`flint-op fwbf16 layer 1 elems 45056 worst_ppm 1441874 ties 0 FAIL` (rc 1). The truncation
mutant failed earlier, `layer 0 ... worst_ppm 4758150 ... FAIL`. Diagnostic builds (throwaway,
not in the tree) counted bit-identical elements per compared segment of 4096 (two tokens):
layer 0 embedding 4096, input norm 4096, attention output 4091, post-attention norm 4093,
MLP output 3410, hidden 3651; layer 1 input norm 3643, attention output 1836, post-attention
norm 2226, MLP output 1277, hidden 1813. The largest absolute attention, MLP and post-norm
difference in layer 0 is one bf16 ulp of its element. The three failing elements are
layer-1 post-attention-norm outputs 1205 and 1394 (token 0) and 2325 (token 1, channel 277),
with `|d|` 1.07 to 1.44 times the budget. Their inputs show the
mechanism, residual-add cancellation:

| element | ref x (layer-0 hidden) | om x | ref attention | om attention | ref post-norm | om post-norm |
|---|---|---|---|---|---|---|
| 1205 | 0.00390625 | 0.0034179688 | -0.0043945312 | -0.0042419434 | -0.15234375 | -0.2578125 |
| 1394 | -0.0107421875 | -0.011230469 | 0.011657715 | 0.01171875 | 0.23730469 | 0.12695312 |
| 2325 | 0.0024414062 | 0.001953125 | 0.0007362366 | 0.00078201294 | 0.9453125 | 0.8125 |

The layer-0 hidden `x` is itself a residual sum `h + mlp` of operands near 0.1 whose result is
near 0.004: a one-ulp flip of an operand (0.000488 near 0.11) lands on `x` as 32 of its own
ulps, and the post-attention norm multiplies it by `w_i / rms`. The error of a residual-stream
element is relative to its operands, not to `|ref_i|`, so no `(1 + |ref|)` constant bounds it,
and propagating a one-ulp flip through chained 2048-term dots worst-case grows without a useful
bound within one layer. No site mismatch was found: the differing-element counts grow from 5
(layer-0 attention) to 686 (layer-0 MLP) to 2260 (layer-1 attention) of 4096, while the largest
absolute attention and MLP differences stay one ulp of their elements.
WolframLanguageEvaluator:
`ulp[x_] := 2^(Floor[Log2[Abs[x]]] - 7); {N[ulp[0.0039], 6], N[(0.00390625 - 0.0034179688)/ulp[0.0039], 6], N[(-0.15234375 + 0.2578125)/(0.0634766817238432426 (1 + 0.15234375)), 6], N[1 - 3410/4096, 6], N[1 - 2226/4096, 6]}`
returned `{0.0000152587890625, 31.999996723200013, 1.4418747364110336, 0.16748046875, 0.45654296875}`
(the third value is the reported worst 1441874 ppm).

## bf16-activation forward arm, v2 criterion (declared 2026-10-30 after v1 failed, before any v2 comparison)

v2 replaces the v1 criterion; it is not a widened v1. It was chosen after seeing the layer 0 and
1 diagnostics above and before computing any of its own quantities on either fixture.

**Criterion.** For every compared row (one token at one compare point: 2048 values, or the
150144 logits), with `ref16` the bf16 fixture, `ref32` the f32 fixture (pinned `b5e4bae2...`,
same layout and token IDs) and `om` the native `fwbf16` value,
`N = ||om - ref16||_2` and `D = ||ref32 - ref16||_2`. **PASS iff `N <= D` for every row and no
value is NaN.** A row with `D = 0` (the exact embedding rows) passes only with `N = 0`.
`worst_ppm` reports `max N / D`. The constant 1 is the only one used and needs no data: the
native bf16 forward must be at least as close to the bf16 reference as the reference computed
without bf16 rounding is. `ops fwbf16` reads both fixtures (`fwbf16 FX16 INVENTORY SHARDS FX32`)
and refuses an `FX32` that is not op 8 with the same token IDs and size.

**What v2 shows and does not show.** It is a discrimination criterion against the f32 referee's
own distance from the bf16 referee, not an error theorem. It detects systematic rounding
errors (a biased rounding mode, a skipped normalization, a routing change neither referee made)
through the whole chain. It does not prove each rounding site: dropping one intermediate rounding
can stay below the dense `ref32 - ref16` difference. Two tokens give attention at most two keys,
so bf16 softmax and `P V` rounding are lightly exercised. The bf16 `F.linear` probe used random
shapes, not the model's exact GEMM shapes (`[INFERENCE]` that dispatch does not change it).

**Declared checks and predictions (v2).**
- `fwbf16` clean: predicted PASS with `N / D < 1` on every row.
- `bf16-truncate`: predicted FAIL at the layer-0 input norm, whose inputs are exact embedding
  rows. There `ref32 - ref16` is the RNE rounding error, rms `1/sqrt 12` ulp, while truncation
  differs from RNE by one ulp on about half the elements, rms `sqrt(1/2)` ulp: predicted
  `N / D` near `sqrt 6 = 2.449`.
- `bf16-skip-final-norm`: predicted FAIL at the final-norm rows.
- Fixture admission as in v1, plus the `FX32` checks above.
- Control `bf16-off` (the plain f64 forward, option b): `om` then approximates `ref32`, so
  `N / D` is predicted near 1 at the layer-0 input norm and the control is recorded, not gated.
- A routed top-6 change in `om` that neither referee made is a FAIL, reported with its layer.

WolframLanguageEvaluator:
`{N[1/Sqrt[12], 6], N[Sqrt[1/2], 6], N[Sqrt[1/2]/(1/Sqrt[12]), 6], Sqrt[1/2]/(1/Sqrt[12])}`
returned `{0.288675134594812921, 0.7071067811865474617, 2.4494897427831780982, Sqrt[6]}`.

## bf16-activation forward arm, v3 criterion (declared 2026-10-30 before any clean v2 or v3 comparison)

v3 replaces v2. v2 was never run on the clean arm; the v1 record and the v2 text above are kept
unedited as history.

**Why v2 is superseded (argument, no clean data).** Let `x` be the exact forward,
`e_om = om - x`, `e16 = ref16 - x`, and take `ref32 = x` to f32 accuracy. Then
`N^2 = M^2 + D^2 - 2 <e_om, e16>` with `M = ||om - ref32||_2`. While the native and the reference
bf16 trajectories share their rounding outcomes, `N` is much smaller than `D`. The v1 counts above
show that they stop sharing them from layer 1 (1277 of 4096 layer-1 MLP outputs identical). With
correlation `rho` and `M` near `D`, `N / D` is near `sqrt(2 (1 - rho))`, which tends to `sqrt 2`
in deep rows. The `bf16-off` control has `om` near `ref32`, so its `N / D` is near 1 in every row.
v2 (`N <= D`) would therefore fail a correct native forward in decorrelated rows and pass
`bf16-off` there, and the same holds for any `N <= s D` or `N / ||ref16|| <= c`: in deep rows
they rank `bf16-off` closer than a correct bf16 forward. Rejecting `bf16-off` needs `M`.

**Diagnostics seen before this declaration.** (1) Referee-only statistics of `ref16 - ref32`
(throwaway build, not in the tree, no native value involved): at the layer-0 input norm the
fresh-rounding energy `F = sum ulp_bf16(ref32_i)^2 / 12` is 0.97 to 1.02 of `D^2`; from layer 1
on it is 0.02 to 0.06 of `D^2`, i.e. propagated noise dominates; from about layer 12 several
token-0 rows carry up to 90 percent of `D^2` in one massive-activation element (participation
ratio 1.2 to 20); at the two anchor rows below the largest single element holds 0.00338 and
0.00325 of `sum ulp_i^2`. (2) The controls `bf16-off` and `bf16-truncate` (gated mutants below)
were run with a statistics-only build of the v3 row sums before this commit:
`bf16-off` anchor `N / D` 0.999997 and 1.000000, `M / D` at most 9.5e-4 in every row and below
`1/b` in some; `bf16-truncate` anchor `N / D` 2.435 and 2.489, `M / D` 1.955 and 2.016, and
`M / D` from 0.50 to 22.8 over all rows. (3) The clean anchor outcome is already known from v1
(all 4096 layer-0 input-norm values bit-identical, same native code path); tier A is
pre-registered against the mutants, not the clean arm. No threshold below uses (2) or (3):
`s_A` comes from the flip bound and `b` from `alpha = 0.01` and one degree of freedom.

**Criterion.** Rows as in v2 (one token at one compare point, 2048 values, or one token's 150144
logits; 496 rows for two tokens); `ops fwbf16 FX16 INVENTORY SHARDS FX32`. Per row:
- `D = 0` (the exact embedding rows): PASS iff `N = 0` and `M = 0`.
- Tier A, anchor rows, the layer-0 input norm of each token: PASS iff `N^2 <= D^2 / 2`.
- Tier B, every row with `D > 0`: PASS iff `1/b <= M / D <= b`, `b = 62898`.
- A NaN anywhere fails. The arm PASSes iff every row passes.
`worst_ppm` is the largest row score in parts per million of its threshold:
`N / (D / sqrt 2)` for tier A, `max(M / D, D / M) / b` for tier B; more than 10^6 is a FAIL.
The report also prints `md_min_ppm`, `md_max_ppm` (`M / D`) and `nd_max_ppm` (`N / D`).

**Tier A derivation.** The anchor is the only compare point computed from exact inputs (bf16
embedding rows) without cancellation: `w_i (x_i rsqrt(mean x^2 + eps))` computed in f32 (the
bf16 weight promotes to f32, `modeling_flint.py` `FlintRMSNorm`) and rounded once to bf16; the
native computes the same product in f64 and rounds once through `bfr` (f64 to f32 to bf16). A
native value can differ from the reference only where the f32 and f64 pre-rounding values
straddle a rounding boundary. Their relative difference is one common factor bounded by
`delta = gamma_2048 / 2 + 5 u_f32` (2048 positive f32 squares summed, then eps, rsqrt and two
products). An element straddles with probability at most `delta |v_i| / ulp_i <= 2^8 delta`
and then contributes `ulp_i^2` to `N^2`, while `D^2` is the fresh RNE energy `sum ulp_i^2 / 12`
(0.97 to 1.02 of it, measured on the referees). So `E[N^2] / D^2 <= 12 2^8 delta = 0.188`
(`N / D` 0.434) at the worst-case `delta`, and 0.0083 (`N / D` 0.091) at the probabilistic
`sqrt(2048) u_f32`. `bf16-off` has `N / D = 1` there and truncation `sqrt 6` (it differs from
RNE by one ulp on half the elements). The threshold `N / D <= 1/sqrt 2` sits above the
worst-case expectation and below both. Realization: with straddle indicators `B_i`, a FAIL
needs `sum B_i ulp_i^2 / sum ulp_i^2 > 0.5 / (12 1.016) = 0.0410`; one straddle adds at most
0.00338, so at least 13 straddles are needed, and Bernstein's inequality with that maximal
weight bounds the per-row false-fail probability by 0.020 at the deterministic worst-case
`delta` and by 4.1e-8 at the probabilistic one (equal weights at the worst case: binomial tail
1.4e-15).

**Tier B derivation.** Null model for a correct native forward: `e_om` and `e16` are identically
distributed Gaussian vectors (sums of many independent roundings by the same mechanism), and
independent in decorrelated rows; positive correlation, as in early rows, only pulls `M / D`
toward 1. Among covariances `Sigma` of `a, b` independent `N(0, Sigma)`, rank 1 gives the
heaviest tail of `|a| / |b|`: proved for isotropic rank `k`, whose moments
`Gamma(k/2+s) Gamma(k/2-s) / Gamma(k/2)^2` decrease in `k`, and found numerically for the mixed
spectra sampled below; the massive-activation rows above are close to rank 1. The band
therefore uses the rank-1 tail as the conservative model: `M / D` is a ratio of two independent
half-normals, `P(M / D > t) = (2/pi) arctan(1/t)`, and the same for `D / M`. A family-wise
false-fail level `alpha = 0.01` over both tails of 494 rows (Bonferroni, no independence
needed) gives `b = cot(pi alpha / (4 494)) = 62898`. Tier B is a gross-error test: it rejects a
row more than 4.8 decades from bf16 noise parity.

WolframLanguageEvaluator:
`{Integrate[f^2,{f,0,1/2}]+Integrate[(1-f)^2,{f,1/2,1}], Integrate[f^2,{f,0,1}], Integrate[1,{f,1/2,1}], Sqrt[(Integrate[1,{f,1/2,1}])/(Integrate[f^2,{f,0,1/2}]+Integrate[(1-f)^2,{f,1/2,1}])], Sqrt[Integrate[f^2,{f,0,1}]/(Integrate[f^2,{f,0,1/2}]+Integrate[(1-f)^2,{f,1/2,1}])]}`
returned `{1/12, 1/3, 1/2, Sqrt[6], 2}` (RNE energy, truncation energy, truncation minus RNE
energy, in ulp^2; truncation `N / D` and `M / D` at the anchor).
`u=2^-24; gam[n_]:=n u/(1-n u); d=gam[2048]/2+5u; {N[gam[2048]], N[d], N[12*2^8*d], N[12*2^8*Sqrt[2048]*u]}`
returned `{0.00012208521548040532, 0.00006134063096407961, 0.1884384183216526, 0.008286407592029853}`.
`g[a_,s_]:=Gamma[a+s]Gamma[a-s]/Gamma[a]^2; {Table[N[g[k/2,0.3]],{k,1,9,2}], Simplify[g[1/2,s]]}`
returned `{{1.7013016167040798, 1.0888330346906112, 1.0452797133029865, 1.0302276854314234, 1.0226586657017234}, (Gamma[1/2 - s]*Gamma[1/2 + s])/Pi}`.
`a=1/100; r=494; b=Cot[Pi a/(4 r)]; {N[b,8], N[1/b,8], N[2 r (2/Pi) ArcTan[1/b],8]}`
returned `{62898.0335046174539468839, 0.00001589874824825148567086119987266738, 0.01}`.
`SeedRandom[834]; n=10^6; tail[l_,t_]:=Module[{k=Length[l],a,b},a=RandomVariate[NormalDistribution[],{n,k}]^2 . l; b=RandomVariate[NormalDistribution[],{n,k}]^2 . l; N[Count[a/b,x_/;x>t^2]/n]]; specs={{1},{1,0.1},{1,0.5},{1,0.01,0.01,0.01,0.01},{1,1}}; Table[{s,tail[s,10],tail[s,100]},{s,specs}]`
returned `{{{1}, 0.063206, 0.006297}, {{1, 0.1}, 0.016694, 0.000185}, {{1, 0.5}, 0.010691, 0.00011}, {{1, 0.01, 0.01, 0.01, 0.01}, 0.010297, 2.*^-6}, {{1, 1}, 0.00968, 0.000108}}`
(`P(|a| / |b| > 10)` and `> 100`; rank 1 is heaviest in every case).
`wm=0.00338; thr=0.5/(12*1.016); bern[p_]:=Module[{t=thr-p},Exp[-t^2/(2 (p wm + wm t/3))]]; {N[thr], N[1/(12 wm)], N[bern[0.0157084]], N[bern[0.008286407592029853/12]], N[1-CDF[BinomialDistribution[2048,0.0157084],85]]}`
returned `{0.04101049868766404, 24.65483234714004, 0.01978754845307135, 4.0618080307047676*^-8, 1.4432899320127035*^-15}`.

**What v3 shows and does not show.** Tier A alone carries the rounding-presence and
rounding-mode verdict: RMSNorm output on exact inputs, both tokens. Tier B checks every row of
the chain, including the logits, for gross divergence only. Neither tier detects any set of
dropped or wrongly rounded sites after the anchor, up to all of them: the `bf16-off` control's
`M / D` lies inside the tier B band in almost every row. With one native and one reference
trajectory, decorrelated deep rows and near-rank-1 massive-activation rows, rounding parity of
deep rows is not decidable at a useful false-fail level. "`ref32 = x` to f32 accuracy" is a
direction, not a number, in deep rows (the f32 arm used 694277 ppm of its budget); the v2
argument does not depend on it. The per-operator bounds of ops 1 to 7 remain the tight local
checks. The v2 limitations (two keys in attention, random-shape `F.linear` probe) still apply.

**Declared checks and predictions (v3).**
- `fwbf16` clean: predicted PASS, anchor `N / D` below 0.434 (most likely near 0, v1 saw all
  4096 anchor values bit-identical), every `M / D` inside the band.
- `bf16-off` (gated mutant: `bfr` is the identity): predicted FAIL at tier A, anchor `N / D`
  near 1 (row statistics already seen, (2) above; the gated verdict is not yet run).
- `bf16-truncate`: predicted FAIL at tier A, anchor `N / D` near `sqrt 6`, `M / D` near 2 (row
  statistics already seen, (2) above; the gated verdict is not yet run).
- `bf16-skip-final-norm`: predicted FAIL at tier B on the token-0 final-norm row, `M / D` near
  `||ref32 layer-48 hidden|| / D = 16557.8 / 0.183 = 90480` (referee norms; 1.44 times `b`).
  The token-1 row is marginal (`18530.9 / 0.287 = 64568`, 1.03 times `b`); the logits rows are
  not predicted numerically (their direction under the mutant depends on `lm_head`).
- Fixture admission: v2's, including an `FX32` that is not op 8 or differs in size or token IDs.
WolframLanguageEvaluator: `{N[16557.8/0.183,6], N[18530.9/0.287,6], N[62898.0335/(16557.8/0.183)], N[Sqrt[0.1884384183216526],6], N[Sqrt[0.008286407592029853],6]}`
returned `{90479.78142076502, 64567.59581881534, 0.6951612007935837, 0.43409494159878503, 0.09102970719512314}`.

**v3 measured (host only, EXERCISED, after declaration commit e9274902).** Full gate
`FLINT_FORWARD=1 bash om/flint/check.sh` on 8a3caf40 plus the check.sh wiring commit:
`OM-FLINT-MANIFEST PASS (100)`, 3m49.6s wall under the shared lock.
- `fwbf16` clean: `layer 49 elems 1312000 worst_ppm 179 ties 6 md_min_ppm 88417 md_max_ppm
  5041815 nd_max_ppm 5208925 PASS`; layer-0 worst_ppm 15 (anchor within tier A by far).
  `M / D` stays in 0.088 to 5.04, inside the band by more than 4 decades.
- `bf16-off`: FAIL at layer 0, worst_ppm 1414212 (anchor `N / D` 1.000, tier A), md 24 to 520 ppm.
- `bf16-truncate`: FAIL at layer 0, worst_ppm 3520459 (anchor `N / D` 2.49), md_min 1.95.
- `bf16-skip-final-norm`: FAIL at layer 49, worst_ppm 1614446 (largest `M / D` 101545, 1.61 `b`).
- Fixture admission: `fwbf16 forward.fx forward.fx`, `forward fwd16.fx`, `fwbf16 fwd16x.fx
  forward.fx` and `fwbf16 fwd16.fx fwd16.fx` (FX32 not op 8) each `refused 7 fixture`.
Every predicted verdict held. The row that carries the skip-final-norm maximum (101545, above the
predicted token-0 final-norm 90480) was not identified; the clean v3 run is the first clean
comparison under v2 or v3, after e9274902.
