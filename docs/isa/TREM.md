# TREM

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

![TREM tile operation](../figures/isa/TREM.svg)

## Introduction

Elementwise remainder of two tiles. For a nonzero divisor, a nonzero result has the same sign as the divisor.

## Math Interpretation

For each element `(i, j)` in the valid region with a nonzero divisor:

$$\mathrm{dst}_{i,j} = \mathrm{remainder}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) = \mathrm{src0}_{i,j} - \mathrm{floor}(\frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}) \times \mathrm{src1}_{i,j}$$

For a nonzero divisor (`src1`), a nonzero result has the same sign as the divisor; exact division gives zero.
For example, `remainder(-7, 3) = 2`, `remainder(7, -3) = -2`, and `remainder(-6, 3) = 0`.
On A5, `int64_t` follows this floor-based definition, with the same remainder semantics as `int32_t` for nonzero divisors.

**Note**: This differs from `TFMOD` where the result sign follows the dividend (`src0`).

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

`PrecisionType` has the following values available:

* `RemAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `RemAlgorithm::HIGH_PRECISION`: High precision algorithm, only supports `float` type.

## High Precision Algorithm (TREM)

`RemAlgorithm::HIGH_PRECISION` runs the same 11-iteration base-$2^{24}$ digit-peeling and the same staged loop inversion as TREMS (see the [TREMS high precision section](TREMS.md#high-precision-algorithm) for the algorithm and the shared tiling budget). The differences all come from the divisor being a tensor:

- **Per-pass divisor reload**: each pass re-reads `|src1|` from the original tile (`vabs` into the scratch `div` region), because a scaled divisor must not feed the next pass.
- **Per-element subnormal regime**: with a scalar divisor, "is the divisor subnormal" is a host-side constant; here iteration 1 decides per element — where `|src1| <= minNormal`, divisor and state are lifted by $2^{24}$ (packed-mask compare + selects) and the post pass rescales exactly those lanes by $2^{-24}$.
- **Exception fixups are per-element selects**: `src1 = ±inf → src0`; then (after it) `src0 = ±inf ∧ src1 = ±inf → NaN`; then `src0 = ±0` with an in-domain divisor `→ src0`.
- **One more scratch region**: the unscaled `|src1|` row persists across a pass's iterations, so TREM's tmp is $192C$-based (784 floats at $C=1$, including the 512-float compensation window; it grows with $\min(C,8)$ for wider rows) versus TREMS's $128C$ — see the `tmpCols` cases equation in [TREMS](TREMS.md#tmp-occupancy).

**Divisor domain**: `src1` elements must be nonzero and not NaN. $\pm\infty$ is fine ($x \bmod \pm\infty = x$). The guaranteed domain is $|y| \ge 1$ (or $\pm\infty$): the remainder sign correction probes $r \cdot y$, which underflows to $\pm 0$ when $|r \cdot y| < 2^{-149}$ and would then miss the correction — a limitation inherited from the A5 implementation. Smaller divisors follow the A5 iteration arithmetic including that limitation.

## Supported Tile Sizes (A2/A3)

TREM carries one more `tmp` region than TREMS (the unscaled `|src1|` row), and the int DEFAULT path routes all float-domain conversions through `tmp` (both input tiles read-only). The `tmp` width per mode, with $R$ the dst row stride in elements, $\mathit{maskF} = 8\lceil R/256\rceil$ and $C = \max(\lfloor N/64 \rfloor, 1)$:

$$
\mathit{tmpCols}(N)=
\begin{cases}
R+8\lceil R/256\rceil+8, & \text{DEFAULT},\ \texttt{float} \\[2pt]
3R+8\lceil R/256\rceil+8, & \text{DEFAULT},\ \texttt{int32\_t} \\[2pt]
272+512, & \text{HIGH\_PRECISION},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+512\min(C,8), & \text{HIGH\_PRECISION},\ C\ge 2
\end{cases}
$$

The $C=1$ HIGH_PRECISION branch folds in the 64-float per-vector state buffer; the $512\min(C,8)$ term is the batched compensation window of the exact final iteration. Widest valid `N` per row count under the 192 KB budget and `validCols/64 <= 255`:

| rows | TREM float | TREM int32 | TREM HP |
|---|---|---|---|
| 1 | 12160 | 8128 | 7424 |
| 2 | 6976 | 5440 | 4928 |
| 4 | 3712 | 3264 | 2944 |
| 8 | 1920 | 1792 | 1664 |
| 16 | 960 | 960 | 832 |
| 32 | 448 | 448 | 448 |

Wider tiles must be split into multiple calls. In-place `dst == src0` is supported in both precision modes (`src1` must remain disjoint).

## Constraints

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `float`, `float32_t`, `int32_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - Tile location must be vector (`TileData::Loc == TileType::Vec`).
    - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
    - In-place operation (`dst` assigned over `src0`) is supported in both precision modes; `src1` must stay disjoint from `dst`.
    - `tmp` must have at least 1 valid row and `TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` valid columns for `float` (intermediate results, a packed comparison mask aligned to 32 bytes, and a 32-byte address buffer), or `3 * TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` for `int32_t` (the int path routes its float-domain dividend, divisor and quotient chain through `tmp`; both input tiles are read-only). The HIGH_PRECISION path needs the larger $192C$ scratch contract from the [TREMS tiling budget](TREMS.md#high-precision-tiling-budget).
- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `half`, `float`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - Tile location must be vector (`TileData::Loc == TileType::Vec`).
    - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
    - Note: `tmp` parameter is accepted but not used on A5.
- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
- **Division-by-zero**:
    - On A5, `int64_t` returns `-1` and `uint64_t` returns `UINT64_MAX` (all 64 bits set) when the divisor is zero.
    - Behavior is target-defined; the CPU simulator asserts in debug builds.
- **High Precision Algorithm**:
    - Available on A5 and A2/A3 for `float` only; staged execution requires `validRows * (validCols / 64) >= 48` and a `dst` tile disjoint from both inputs, with `validCols / 64 <= 255` (split wider tiles).
    - Divisors must be nonzero and not NaN; the guaranteed domain is `|src1| >= 1` or `±inf` (see [High Precision Algorithm (TREM)](#high-precision-algorithm-trem)).

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 32>;
  TileT dst, src0, src1;
  TmpT tmp;
  TREM(dst, src0, src1, tmp);

  // High precision (float only): tmp sized per the TREMS tiling budget,
  // e.g. 8x448 tiles -> C = 7, tmpCols = 192*7 + 8*ceil(7/4) + 8 + 512*7 = 4952
  Tile<TileType::Vec, float, 8, 448> hpDst, hpSrc0, hpSrc1;
  Tile<TileType::Vec, float, 1, 4952> hpTmp;
  TREM<RemAlgorithm::HIGH_PRECISION>(hpDst, hpSrc0, hpSrc1, hpTmp);
}
```
