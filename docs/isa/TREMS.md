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

![TREMS tile operation](../figures/isa/TREMS.svg)

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

Two execution organizations exist (issue #2 loop inversion). The per-vector path runs all 11 iterations on one 64-element vector at a time through the scratch state buffer — the fallback for small tiles, aliased tiles and row tails. The staged path advances **every full vector by one iteration per pass** with the per-vector state staged in `dst`; on A2/A3 each op of a pass is one repeat instruction covering all full vectors of a row:

```
seed:      dst[row] = |src[row]|                     (vabs, rep = fullPerRow)
iters 11→1: for each row: SolveScale iteration         (ops with rep = fullPerRow,
             on dst[row] with state in place            state = dst[row])
final:     iteration 1 runs in chunks of min(8, C) vectors (the
           compensated subtract's window scales with the chunk) state = dst[row])
post:      for each row: HpPostS on dst[row]
tails:     per-vector fallback on the <64-element tail of each row
```

Staging engages when `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS` (48) and the `dst` tile does not alias the input; `validCols / 64 <= 255` (8-bit repeatTimes) always applies — split wider tiles into multiple calls.

### Exact final iteration

The last peel iteration computes `state -= k * b` with single-rounding semantics. A5 evaluates it as one fused `vmula`; A2/A3 has no fused multiply-add (`vmla` rounds the product first), so the port splits $k$ (an integer below $2^{24}$) and $b$ into 12-bit halves, evaluates the four exact partial products, reassembles $k \cdot b$ exactly as a `(hi, lo)` pair with full TwoSum chains, and subtracts both parts. The first subtraction is Sterbenz-exact, so the second rounds once and yields the true remainder — which is always representable. The staged final pass batches $\min(C, 8)$ vectors per call (`TREM_HP_FUSED_CHUNK_A2A3`), converging the exactness overhead to roughly +35% vector-core cycles for tiles up to 512 columns. Lanes whose $|b|$ would overflow the split ($|b| > \text{FLT\_MAX}/4097$) keep the plain two-op result, where the error stays within half an ulp of the dividend (the tolerated class for quotients beyond $2^{24}$). This is what makes results such as $4580025 \bmod -7.25$ exact ($-3.0$, not $-3.25$).

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
| `comp` | `+512\,\min(C,8)$ | final-iteration compensated subtract (`HpFusedSub`), batched over $\min(C,8)$ vectors per call: 12-bit split halves, four exact partial products and the TwoSum reassembly of $k \cdot b$ |

The quotient/temp sharing is safe because `k` dies at the `vmul` that reads it, and the post-pass probe is consumed by `vcmp` before the candidate write — every handoff is separated by a `pipe_barrier` (in-place `dst == src0` ops are the established A2/A3 idiom). The staged path keeps state in `dst`, so `state` never scales with $C$.

### tmp occupancy

$$
\mathit{tmpCols}(N)=
\begin{cases}
208+512, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8+512\min(C,8), & \text{TREMS},\ C\ge 2 \\[2pt]
272+512, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+512\min(C,8), & \text{TREM},\ C\ge 2
\end{cases}
\qquad
C=\max\Bigl(\Bigl\lfloor \frac{N}{64}\Bigr\rfloor,\ 1\Bigr)
$$

The $8\lceil C/4\rceil$ term is the packed compare mask (32-byte padded); the constant $C=1$ branch folds in the 64-float per-vector state buffer, and the $+512\min(C,8)$ term is the batched compensation window of the final iteration's exact `state -= k*b` (see below; `TREM_HP_FUSED_CHUNK_A2A3` = 8). TREM carries one more region than TREMS — the unscaled `|src1|` row must persist across a pass's iterations as the rescaling source (it cannot live in `bTmp`, which iteration 1 overwrites, nor in `k`, which the quotient overwrites mid-iteration).

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
| $[1, 64]$ | 256 | 256 | 2880 | 3392 | 2% | per-vector (1) |
| $[32, 64]$ | 8192 | 8192 | 2880 | 19264 | 10% | per-vector (32) |
| $[12, 256]$ | 12288 | 12288 | 10304 | 34880 | 18% | staged (48) |
| $[8, 448]$ | 14336 | 14336 | 18016 | 46688 | 24% | staged (56) |
| $[2, 2048]$ | 16384 | 16384 | 33056 | 65824 | 33% | staged (64) |
| $[1, 8192]$ | 32768 | 32768 | 82976 | 148512 | 76% | staged (128) |
| $[8, 2496]$ | 79872 | 79872 | 36704 | 196448 | 100% | staged (312) |
| $[1, 11136]$ | 44544 | 44544 | 106912 | 196000 | 100% | staged (174) |

Largest packed `W` per row count (TREMS; rule of thumb $W \lesssim 22278/(R+1)$):

| rows R | 1 | 2 | 3 | 4 | 7 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| max W | 11136 | 7424 | 5568 | 4480 | 2752 | 2496 | 1728 | 1280 | 640 |

For TREM ($W \lesssim 14852/(R+1)$): max W = 7424 / 4928 / 3712 / 2944 / 1856 / 1664 / 1152 / 832 / 448 for the same row counts; the widest legal single row is `[1, 7424]` (195520 B).

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
| 1 | 16192 | 12160 | 11136 |
| 2 | 9728 | 8128 | 7424 |
| 4 | 5440 | 4864 | 4480 |
| 8 | 2880 | 2688 | 2496 |
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
  // (C = 448/64 = 7, tmpCols = 128*7 + 8*ceil(7/4) + 8 + 512*7 = 4504)
  using HpTmpT = Tile<TileType::Vec, float, 1, 4504>;
  Tile<TileType::Vec, float, 8, 448> hpOut, hpSrc;
  HpTmpT hpTmp;
  TREMS<RemSAlgorithm::HIGH_PRECISION>(hpOut, hpSrc, 7.5f, hpTmp);
}
```
