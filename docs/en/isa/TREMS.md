# TREMS

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->



## Tile Operation Diagram

![TREMS tile operation](../../figures/isa/TREMS.svg)

## Introduction

Elementwise remainder with a scalar. For a nonzero scalar divisor, a nonzero result has the same sign as the divisor.
`RemSAlgorithm::HIGH_PRECISION` selects a high precision FP32 algorithm (see [High Precision Algorithm](#high-precision-algorithm)).

## Math Interpretation

For each element `(i, j)` in the valid region with a nonzero divisor:

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{floor}(\frac{\mathrm{src}_{i,j}}{\mathrm{scalar}}) \times \mathrm{scalar}$$

For a nonzero `scalar`, a nonzero result has the same sign as `scalar`; exact division gives zero.
For example, `remainder(-7, 3) = 2`, `remainder(7, -3) = -2`, and `remainder(-6, 3) = 0`.
On A5, `int64_t` follows this floor-based definition, with the same remainder semantics as `int32_t` for nonzero divisors.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events);
```

`PrecisionType` has the following values available:

* `RemSAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `RemSAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower, only supports `float` type.

## High Precision Algorithm

The high precision mode computes the same floored remainder, but resolves it to full `float32` precision instead of degrading once $|x/s| > 2^{24}$ (the one-shot $q \cdot s$ product of the default path rounds at the magnitude of $x$, and the subtraction cancels catastrophically, leaving $\pm(1\ldots 2)\,\mathrm{ulp}(x)$ — or $\pm\infty$ at the `FLT_MAX` boundary). Special values follow IEEE (`x % s` in numpy semantics):

| x | s | result |
|---|---|--------|
| $\pm\infty$ | finite | NaN |
| finite | $\pm\infty$ | x (NaN if x is also $\pm\infty$) |
| $\pm 0$ | $s \ne 0$ | $\pm 0$ (sign of x) |
| NaN | any | NaN |

The algorithm is base-$2^{24}$ digit-peeling long division. Work on $\mathit{state} = |x|$ and $\mathit{div} = |s|$; the f32 quotient range spans $\sim 2^{276}$, and the exponent gap is peeled from the top in ~24-bit chunks by 11 iterations ($\lceil 276/24 \rceil = 11$):

| iteration (runs high→low) | divisor scale $\mathit{div} \cdot 2^{e}$ |
|---|---|
| 11 | $2^{240}$ |
| 10 | $2^{216}$ |
| 9 | $2^{192}$ |
| 8 | $2^{168}$ |
| 7 | $2^{144}$ |
| 6 | $2^{120}$ |
| 5 | $2^{96}$ |
| 4 | $2^{72}$ |
| 3 | $2^{48}$ |
| 2 | $2^{24}$ |
| 1 | $2^{0}$ (unscaled) + subnormal regime |

$$
\begin{aligned}
b &= \mathit{div} \cdot 2^{e(i)} \\[2pt]
k &= \begin{cases} \lfloor \mathit{state}/b \rfloor & \text{remainder (\texttt{ROUND\_F}}) \\[2pt]
      \operatorname{trunc}(\mathit{state}/b) & \text{fmod (\texttt{ROUND\_Z}}) \end{cases} \\[2pt]
\mathit{state} &\leftarrow \mathit{state} - k \cdot b
\end{aligned}
$$

$b$ is an exact power-of-two scaling of the divisor; $k$ is an integer and therefore exact in f32 (the int32 conversion saturates, a free quotient clamp at extreme exponent gaps); the subtraction cancels the top ~24 bits of the gap with at most 1 ulp of rounding at the chunk's magnitude — never at $|x|$'s. Iteration 1 additionally lifts divisor and state by $2^{24}$ when $|s| \le \mathit{minNormal}$ (the post pass rescales by $2^{-24}$), takes the final unscaled $k = \lfloor \mathit{state}/|s| \rfloor$, and applies a subnormal-scale sign fix. The post pass restores the dividend sign, applies the exception fixups above, and corrects the remainder sign ($\mathit{state} \cdot s < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + s$).

A rigorous mathematical specification of the algorithm is in the [appendix](#section_hp_appendix).

Two execution organizations exist (issue #2 loop inversion). The per-vector path runs all 11 iterations on one 64-element vector at a time through the scratch state buffer — the fallback for small tiles, aliased tiles and row tails. The staged path advances **every full vector by one iteration per pass** with the per-vector state staged in `dst`; on A2/A3 each op of a pass is one repeat instruction covering all full vectors of a row:

```
seed:      dst[row] = |src[row]|                     (vabs, rep = fullPerRow)
iters 11→1: for each row: SolveScale iteration         (ops with rep = fullPerRow,
             on dst[row] with state in place            state = dst[row])
final:     iteration 1 runs in chunks of min(8, C) vectors (the
           compensated subtract's window scales with the chunk)
post:      for each row: HpPostS on dst[row]
tails:     per-vector fallback on the <64-element tail of each row
```

Staging engages when `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS` (48) and the `dst` tile does not alias the input; `validCols / 64 <= 255` (8-bit repeatTimes) always applies — split wider tiles into multiple calls.

### Exact final iteration

The last peel iteration computes `state -= k * b` with single-rounding semantics. A5 evaluates it as one fused `vmula`; A2/A3 has no fused multiply-add (`vmla` rounds the product first), so the port splits $b$ with the Veltkamp constant $257 = 2^8+1$ ($b_H \le 8$ significant bits, $b_H + b_L = b$ exactly) and the quotient digit $k < 2^{24}$ into three 8-bit digits $k_1 2^{16} + k_2 2^8 + k_3$. Every partial product $k_i \cdot b_{\{H,L\}}$ holds $\le 24$ bits and is exact; a cascade of standard TwoSums reassembles $k \cdot b$ exactly as a `(hi, lo)` pair, and subtracting both parts rounds once and yields the true remainder — which is always representable. The staged final pass batches $\min(C, 8)$ vectors per call (`TREM_HP_FUSED_CHUNK_A2A3`). A result-finiteness guard ($|r| \le \text{FLT\_MAX}$; NaN compares false) routes the rare near-FLT_MAX overflow lanes and the $|b| > \text{FLT\_MAX}/257$ region (where the split itself overflows) to the plain two-op result, whose error stays within half an ulp of the dividend. This is what makes results such as $4580025 \bmod -7.25$ exact ($-3.0$, not $-3.25$), keeps $\text{FLT\_MAX} \bmod \text{FLT\_MAX}/2^{23}$ exact ($0$, not NaN), and recovers the true remainder for rich-mantissa divisors with $|b|$ up to $\text{FLT\_MAX}/257$.

Infinite divisors follow floored semantics (numpy `\%`): a finite $x$ with $\operatorname{sign}(x) = \operatorname{sign}(y)$ returns $x$, with opposite signs it returns $\operatorname{copysign}(\infty, y)$, $x = \pm\infty$ returns NaN, and $x = \pm 0$ keeps $x$.

## High Precision Tiling Budget

### tmp tile layout

The HP scratch is a capacity-sized window over the `tmp` tile, $C = \max(\lfloor N/64 \rfloor, 1)$ vectors wide ($N$ = valid columns). Buffers with disjoint lifetimes share storage:

| region | floats | lifetime |
|---|---|---|
| `bTmp` | `[0, 64C)` | within one iteration — scaled divisor, `vector_dup`'d per iteration from host-computed scalars |
| `k` | `[64C, 128C)` | quotient; after the product `vmul` also the general temp (sign factor, select operands, probe, correction candidate) |
| `cmpMask` | `[128C, +maskF)` | packed compare bitmask, from `vcmp` until the `vsel` that consumes it |
| `addrBuf` | `+8` | the full 64-bit `cmpMask` address, around each `set_cmpmask` |
| `state` | `+64`, only when $C = 1$ | per-vector fallback state across the 11 iterations |
| `comp` | `+640\,\min(C,8)$ | final-iteration compensated subtract (`HpFusedSub`), batched over $\min(C,8)$ vectors per call: 8-bit Veltkamp split of $b$, three 8-bit quotient digits, six exact partial products and the TwoSum reassembly of $k \cdot b$ |

The quotient/temp sharing is safe because `k` dies at the `vmul` that reads it, and the post-pass probe is consumed by `vcmp` before the candidate write — every handoff is separated by a `pipe_barrier` (in-place `dst == src0` ops are the established A2/A3 idiom). The staged path keeps state in `dst`, so `state` never scales with $C$.

### tmp occupancy

$$
\mathit{tmpCols}(N)=
\begin{cases}
208+640, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8+640\min(C,8), & \text{TREMS},\ C\ge 2 \\[2pt]
272+640, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+640\min(C,8), & \text{TREM},\ C\ge 2
\end{cases}
\qquad
C=\max\Bigl(\Bigl\lfloor \frac{N}{64}\Bigr\rfloor,\ 1\Bigr)
$$

The $8\lceil C/4\rceil$ term is the packed compare mask (32-byte padded); the constant $C=1$ branch folds in the 64-float per-vector state buffer, and the $+640\min(C,8)$ term is the batched compensation window of the final iteration's exact `state -= k*b` (see below; `TREM_HP_FUSED_CHUNK_A2A3` = 8). TREM carries one more region than TREMS — the unscaled `|src1|` row must persist across a pass's iterations as the rescaling source (it cannot live in `bTmp`, which iteration 1 overwrites, nor in `k`, which the quotient overwrites mid-iteration).

### Per-tile occupancy and maximum tiling

With $n_{\text{tiles}} = 2$ (TREMS) or 3 (TREM) data tiles of $M \times N$:

$$
\mathit{UBUF}(M,N) \;=\;
\underbrace{n_{\text{tiles}}\cdot 4MN}_{\text{data tiles}} \;+\;
\underbrace{4\,\mathit{tmpCols}(N)}_{\text{tmp}}
\;\le\; 196608\ \mathrm{B}, \qquad C \le 255
$$

Worked examples — byte size of each tile:

| tile $[M, N]$ | dst | src | tmp | total UBUF | of 196608 B | dispatch ($M \cdot C$ vs 48) |
|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 3392 | 3904 | 2% | per-vector (1) |
| $[32, 64]$ | 8192 | 8192 | 3392 | 19776 | 10% | per-vector (32) |
| $[12, 256]$ | 12288 | 12288 | 12352 | 36928 | 19% | staged (48) |
| $[8, 448]$ | 14336 | 14336 | 21600 | 50272 | 26% | staged (56) |
| $[2, 2048]$ | 16384 | 16384 | 37152 | 69920 | 36% | staged (64) |
| $[1, 8192]$ | 32768 | 32768 | 87072 | 152608 | 78% | staged (128) |
| $[8, 2432]$ | 9728 | 9728 | 40288 | 195936 | 100% | staged (304) |
| $[1, 10880]$ | 43520 | 43520 | 108928 | 195968 | 100% | staged (170) |

Largest packed `W` per row count (TREMS; rule of thumb $W \lesssim 21760/(R+1)$):

| rows R | 1 | 2 | 3 | 4 | 7 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| max W | 10880 | 7296 | 5440 | 4352 | 2688 | 2432 | 1664 | 1280 | 640 |

For TREM ($W \lesssim 14400/(R+1)$): max W = 7296 / 4864 / 3648 / 2880 / 1792 / 1600 / 1088 / 832 / 448 for the same row counts; the widest legal single row is `[1, 7296]` (196544 B).

Design checklist:
- Pick the working width $N$ first — it alone fixes the tmp size; then bound the rows by $M \le (196608 - 4\,\mathit{tmpCols}(N))\,/\,(4N \cdot n_{\text{tiles}})$.
- $C \le 255$ and $M \cdot C \ge 48$ if you want the staged path.
- Column-block anything larger: each call processes one tile.
- The DEFAULT path's tmp contract is much smaller (`dstRowStride + maskF + 8` floats), so a tile can be legal for DEFAULT and illegal for HIGH_PRECISION; check the budget for the precision mode you dispatch.

## Supported Tile Sizes (A2/A3)

The DEFAULT `tmp` width, with $R$ the dst row stride in elements and $\mathit{maskF} = 8\lceil R/256\rceil$ (the int path keeps its float-domain dividend and quotient chain in `tmp` — input tiles are read-only):

$$
\mathit{tmpCols}_{\text{DEFAULT}}(N)=
\begin{cases}
R+8\lceil R/256\rceil+8, & \texttt{float} \\[2pt]
2R+8\lceil R/256\rceil+8, & \texttt{int32\_t}
\end{cases}
$$

The HIGH_PRECISION path needs the larger contract from the [tiling budget](#high-precision-tiling-budget) above. With the 192 KB unified-buffer budget and `validCols/64 <= 255` (8-bit repeat count), the widest valid `N` per row count is:

| rows | TREMS float | TREMS int32 | TREMS HP |
|---|---|---|---|
| 1 | 16192 | 12160 | 10880 |
| 2 | 9728 | 8128 | 7296 |
| 4 | 5440 | 4864 | 4352 |
| 8 | 2880 | 2688 | 2432 |
| 16 | 1472 | 1408 | 1280 |
| 32 | 704 | 704 | 640 |

Rule of thumb: `N \lesssim 23500/(rows+1)` (float), `N \lesssim 17500/(rows+1)` (int32), and the staged-HP caps from the budget section. Wider tiles must be split into multiple calls. In-place `dst == src` is supported in both precision modes.

## Constraints

- **Implementation Checks (A2A3)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst` and `src` must be vector tiles.
    - `dst` and `src` must be row-major.
    - Runtime: `dst.GetValidRow() == src.GetValidRow() > 0` and `dst.GetValidCol() == src.GetValidCol() > 0`.
    - In-place operation (`dst` assigned over `src`) is supported in both precision modes.
    - **tmp Buffer Requirements**:
      - `tmp.GetValidCol() >= dstRowStride + maskF + 8` for `float`, or `>= 2*dstRowStride + maskF + 8` for `int32_t` (the int path keeps its float-domain dividend and quotient chain in `tmp`; input tiles are read-only), where `maskF = 8*ceil(dstRowStride/256)`
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - Data type must match `TileDataDst::DType`.
      - The HIGH_PRECISION path needs the larger scratch contract from the [High Precision Tiling Budget](#high-precision-tiling-budget) section.
- **Implementation Checks (A5)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst` and `src` must be vector tiles.
    - Static valid bounds: `ValidRow <= Rows` and `ValidCol <= Cols` for both tiles.
    - Runtime: `dst.GetValidRow() == src.GetValidRow()` and `dst.GetValidCol() == src.GetValidCol()`.
    - Note: tmp parameter is accepted but not validated or used on A5.
- **Division by Zero**:
    - On A5, `int64_t` returns `-1` and `uint64_t` returns `UINT64_MAX` (all 64 bits set) when `scalar` is zero.
    - Behavior is target-defined; the CPU simulator asserts in debug builds.
- **Valid Region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
- **For `int32_t` Inputs (A2A3 Only)**: Both `src` elements and `scalar` must be in the range `[-2^24, 2^24]` (i.e., `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.
- **High Precision Algorithm**:
    - Available on A5 and A2/A3 for `float` only; see [High Precision Algorithm](#high-precision-algorithm).
    - Staged execution requires `validRows * (validCols / 64) >= 48` and a `dst` tile disjoint from the input; smaller or aliased tiles use the per-vector fallback.
    - `validCols / 64 <= 255` (one row-batch repeat instruction covers at most 255 full vectors); split wider tiles into multiple calls.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);

  // High precision: 8x448 tile, tmp sized per the High Precision Tiling Budget
  // (C = 448/64 = 7, tmpCols = 128*7 + 8*ceil(7/4) + 8 + 640*7 = 5400)
  using HpTmpT = Tile<TileType::Vec, float, 1, 5400>;
  Tile<TileType::Vec, float, 8, 448> hpOut, hpSrc;
  HpTmpT hpTmp;
  TREMS<RemSAlgorithm::HIGH_PRECISION>(hpOut, hpSrc, 7.5f, hpTmp);
}
```

## Appendix: High-Precision Algorithm Specification (A2/A3)<a name="section_hp_appendix"></a>

This appendix is the mathematical specification of the current A2/A3 `HIGH_PRECISION` implementation. Let $\mathrm{RN}$ denote `float32` round-to-nearest (24-bit significand, $\mathit{minNormal} = 2^{-126}$, $\mathit{FLT\_MAX} = (2 - 2^{-23})\,2^{127}$), and $\operatorname{ulp}(v)$ the unit in the last place of $v$. Both operands of every inner product below are nonnegative.

### A.1 Semantics and domain

For finite $y \ne 0$ the result is the floored remainder

$$
r \;=\; x - \lfloor x/y \rfloor\, y, \qquad r\,\operatorname{sign}(y) \in [\,0,\ |y|\,), \qquad \operatorname{sign}(r) = \operatorname{sign}(y)\ \text{for}\ r \ne 0,
$$

computed to full `float32` precision — exact whenever the exact value is representable (always, by A.3). Contract: $y \ne 0$ and $y \ne \mathrm{NaN}$. Guaranteed envelope: $|y| \ge 1$ or $y = \pm\infty$. Special values:

| $x$ | $y$ | result |
|---|---|---|
| NaN | any | NaN |
| $\pm\infty$ | any | NaN |
| $\pm 0$ | $y \ne 0$ | $x$ (keeps the sign of $x$) |
| finite | $\pm\infty$ | $x$ if $\operatorname{sign}(x) = \operatorname{sign}(y)$, else $\operatorname{copysign}(\infty, y) = x + y$ |

### A.2 Digit peeling

The core reduces $\mathit{state} = |x|$ against $\mathit{div} = |y|$ one 24-bit digit at a time. Iterations $i = 11, \dots, 1$ use the scale ladder $e(i) = 24(i-1)$, applied as at most two exact power-of-two scalings (the two-step form avoids intermediate overflow; TREMS folds it into the host constant $\mathrm{RN}(\mathrm{RN}(|y| 2^{a}) 2^{b})$, TREM applies it on the unit):

$$
b_i \;=\; \mathit{div}\cdot 2^{e(i)}, \qquad
k_i \;=\; \begin{cases} \lfloor \mathit{state}/b_i \rfloor & \text{remainder (round-to-nearest-int)} \\ \operatorname{trunc}(\mathit{state}/b_i) & \text{fmod} \end{cases} \qquad
\mathit{state} \;\leftarrow\; \mathit{state} - k_i\, b_i .
$$

Properties of each step as implemented:

- **Scaled divisor.** $b_i$ is an exact power-of-two scaling of $\mathit{div}$ (exact in `float32` unless it overflows). On non-final iterations $b_i$ is clamped to $\mathit{FLT\_MAX}$ (`vmins`), so a ladder rung with $b_i > \mathit{state}$ produces $k_i = 0$ and the product $0 \cdot b_i$ stays $0$ even when the unclamped $b_i$ would be $+\infty$.
- **Quotient digit.** $k_i = \mathrm{roundToInt}(\mathrm{RN}(\mathit{state}/b_i))$ through a saturating `f32 → s32 → f32` conversion: every integer $|k_i| \le 2^{24}$ round-trips exactly; larger quotients (reachable only for $|y| < 1$, outside the envelope) saturate at $2^{31}$, one rung clamping the exponent gap by $\ge 31$ bits per pass. Because consecutive rungs are spaced $24$ exponent bits apart, rung $i+1$ leaves $\mathit{state} < b_{i+1}\cdot(1 + O(2^{-23}))$, so each subsequent digit satisfies $k_i \le 2^{24}$ and is exact.
- **Subtractive update.** For iterations $i \ge 2$, $\mathit{state} \leftarrow \mathrm{RN}(\mathit{state} - \mathrm{RN}(k_i b_i))$: both products and the subtract round at the *chunk's* magnitude, never at $|x|$'s, so each iteration cancels the top $\sim 24$ bits of the remaining exponent gap and the accumulated error decays geometrically down the ladder. Iteration $i = 1$ uses the compensated update of A.3, then applies the guard fixup $\mathit{state} < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + b_1$, restoring $\mathit{state} \in [0, b_1)$.

After iteration 1 the state is the exact floored remainder of $|x|$ over $|y|$ (over $|y|\,2^{24}$ in the subnormal regime of A.5), carrying only the final iteration's single rounding — which A.3 removes.

### A.3 Exact final iteration (`HpFusedSub`)

The last update must subtract $k \cdot b$ with **single-rounding** semantics: $\mathrm{RN}(\mathit{state} - k\,b)$ where $k\,b$ is formed exactly. The vector unit has no fused multiply-add, so the product is reconstructed exactly from pieces that each fit the 24-bit significand. Write $s = \mathit{state}$, $b = b_1$, digit $k < 2^{24}$ (integer, exact).

**Lemma 1 (Veltkamp split, $C = 257 = 2^8 + 1$).** With $t = \mathrm{RN}(bC)$, $q = \mathrm{RN}(t - b)$, $b_H = \mathrm{RN}(t - q)$, $b_L = \mathrm{RN}(b - b_H)$:

$$
b_H + b_L = b \ \text{exactly}, \qquad b_H \ \text{has} \le 8 \ \text{significant bits}, \qquad b_L \ \text{has} \le 16,
$$

valid whenever $|b| \le \mathit{FLT\_MAX}/257$ (so $t$ does not overflow). The two subtractions are exact by Sterbenz-type cancellation ($t - b$ cancels the top bits of $t$, leaving the low part of the split; $b - b_H$ is the complementary low part and fits the significand).

**Lemma 2 (digit decomposition).** Split $k = d_1 2^{16} + d_2 2^{8} + d_3$ with $0 \le d_j < 2^8$. The extraction — scale by $2^{-16}$, round to integer, subtract; repeat at $2^{-8}$ — is exact: every intermediate is an integer $< 2^{25}$, and all such subtractions/differences are integers $< 2^{24}$, hence representable.

**Lemma 3 (exact partial products).** Each of the six products $d_j \cdot b_H$ and $d_j \cdot b_L$ ($j = 1,2,3$) has at most $8 + 8 = 16$ or $8 + 16 = 24$ significant bits, hence is computed exactly by `vmul`; the weight factors $2^{16}$, $2^{8}$ are applied after the multiply as exact power-of-two scalings, with $k\,b \le s \le \mathit{FLT\_MAX}$ bounding every intermediate against overflow.

**Lemma 4 (TwoSum).** For any `float32` $a, b$: with $s = \mathrm{RN}(a+b)$, $z = \mathrm{RN}(s-a)$, $e = \mathrm{RN}(b - z)$, the residual $e$ is exact and $s + e = a + b$ exactly (Knuth's 6-op expansion). The six exact partials are streamed through this cascade into a pair $(\mathit{hi}, \mathit{lo})$ with

$$
\mathit{hi} + \mathit{lo} \;=\; k \cdot b \quad \text{exactly}.
$$

**Lemma 5 (Sterbenz subtract).** $r_1 = \mathrm{RN}(s - \mathit{hi})$ is exact: $k \ge 1$ gives $\mathit{hi} \approx kb \in (s/2,\ s]$ so $s/2 < \mathit{hi} \le s$ — the Sterbenz region $y/2 \le x \le 2y$ where subtraction is exact; $k = 0$ gives $\mathit{hi} = 0$, trivially exact.

**Result.** $r = \mathrm{RN}(r_1 - \mathit{lo})$: the exact value $r_1 - \mathit{lo} = s - kb$ is the floored remainder of two `float32` values, and the classical binary-`fmod` representability property (the remainder is $s$ with its leading bits cancelled, so it needs no more than 24 significand bits) makes it representable; $\mathrm{RN}$ of a representable value is the identity, so $r$ is the **exact** remainder.

### A.4 Finiteness guard and fallback

Alongside $r$, the plain two-op value $f_b = \mathrm{RN}(s - \mathrm{RN}(k\,b))$ is computed, and the lane result is

$$
\text{result} \;=\; \begin{cases} r, & |r| \le \mathit{FLT\_MAX} \\ f_b, & \text{otherwise (including } r = \mathrm{NaN}\text{)} \end{cases}
$$

selected by one packed compare (`|r| <= FLT_MAX`; NaN compares false). This routes two lane families to $f_b$: (i) $|b| > \mathit{FLT\_MAX}/257$, where Lemma 1's split is unavailable (the partials are NaN/inf), and (ii) the vanishingly rare near-$\mathit{FLT\_MAX}$ remainder whose compensated form overflows. Since $k\,b \le s \le \mathit{FLT\_MAX}$, $f_b$ is always finite with absolute error $\le \tfrac{1}{2}\operatorname{ulp}(k\,b) \le \tfrac{1}{2}\operatorname{ulp}(s)$.

### A.5 Subnormal-divisor regime

When $|y| \le \mathit{minNormal}$ (decided on the host for TREMS, per element via packed-mask compare+select for TREM), iteration 1 lifts both operands by $2^{24}$ before taking $b_1 = |y|\,2^{24}$: the divisor lift is exact (power-of-two scaling of a subnormal), and the state lift is exact while $|x|\,2^{24} \le \mathit{FLT\_MAX}$. The post pass rescales exactly those lanes by $2^{-24}$; the rescale is exact while the result is normal. This regime lies outside the guaranteed envelope of A.1 and follows the A5 iteration arithmetic.

### A.6 Post pass

Ordered select chain, applied to the magnitude-domain state (vector form; the TREMS scalar form resolves the divisor-dependent branches on the host):

1. **Sign restore**: $x < 0 \Rightarrow \mathit{state} \leftarrow -\mathit{state}$. Negative subnormals compare correctly; $-0$ reads as $0$ and is overwritten by step 4.
2. **Subnormal rescale**: $\mathit{state} \leftarrow \mathit{state}\cdot 2^{-24}$ on the lifted lanes (A.5).
3. **Infinite divisor** ($|y| = \pm\infty$): floored semantics of A.1 — same-sign $x$ keeps $x$; opposite-sign $x$ takes $x + y = \operatorname{copysign}(\infty, y)$; $x = \pm\infty$ becomes NaN (re-applied after the same-sign select, which would otherwise leave the dividend in place). The sign probes multiply by the divisor and $\pm 0 \cdot \pm\infty = \mathrm{NaN}$, so zero lanes keep $x$. In the scalar (TREMS) form an infinite divisor short-circuits the whole iteration chain (the per-iteration divisor broadcasts and the sign probe both misbehave at $\pm\infty$); in the vector (TREM) form the lanes flow through the chain and the floored-inf select runs after step 5, followed only by the NaN re-application.
4. **Zero dividend**: $x = \pm 0$ with an in-domain divisor $\Rightarrow x$ (sign of $x$).
5. **Remainder sign correction**: $\mathit{state} \cdot y < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + y$. Skipped in the scalar form for an infinite $y$; in the vector form it computes (harmlessly overflowing on inf-divisor lanes) before step 3's select overwrites those lanes.

### A.7 Execution organizations

Two schedules evaluate the identical per-lane dataflow graph of A.2–A.6: the **per-vector** path (all 11 iterations + post on one 64-element vector; used for small tiles, aliased tiles, and row tails) and the **staged** path (one iteration per pass over all full vectors of a row with the state staged in `dst`, the final pass batched $\min(C, 8)$ vectors per compensated call). Batching reorders operations *across* lanes only; within a lane the RN dag is unchanged, so both schedules — and any chunking of the staged passes — produce bitwise-identical per-lane results.
