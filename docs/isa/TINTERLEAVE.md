# TINTERLEAVE


## Tile Operation Diagram

![TINTERLEAVE](../figures/isa/TINTERLEAVE.svg)


## Introduction

Interleave two source tiles (`src0` and `src1`) into two destination tiles (`dst0` and `dst1`). The operation combines elements from `src0` and `src1` in an alternating pattern: even-positioned elements of the interleaved stream are placed into `dst0`, and odd-positioned elements into `dst1`. Each destination tile holds half the interleaved stream, split at the midpoint.

`TINTERLEAVE` is the inverse of `TDEINTERLEAVE`.

## Math Interpretation

### Two-source form

Given two source tiles `src0` and `src1` with the same valid shape `(validRows, validCols)`, construct an interleaved stream of length `2 × validCols` per row:

$$ \mathrm{interleaved}_{2k} = \mathrm{src0}_{i, k}, \quad \mathrm{interleaved}_{2k+1} = \mathrm{src1}_{i, k}, \quad 0 \le k < \mathrm{validCols} $$

Then split the interleaved stream into two halves:

$$ \mathrm{dst0}_{i, j} = \mathrm{interleaved}_{j}, \quad 0 \le j < \mathrm{validCols} $$
$$ \mathrm{dst1}_{i, j} = \mathrm{interleaved}_{\mathrm{validCols} + j}, \quad 0 \le j < \mathrm{validCols} $$

Where `validRows = dst0.GetValidRow()` and `validCols = dst0.GetValidCol()`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TInterleave(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0,
                                 WaitEvents &...events);
```

> **Note**: The parameter order is `(dst1, dst0, src1, src0)`. `dst0` receives the first half of the interleaved stream (positions `0 … validCols-1`), `dst1` receives the second half (positions `validCols … 2×validCols-1`).

## Constraints

- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `uint8_t`, `int8_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - All tiles (`dst0`, `dst1`, `src0`, `src1`) must have the same `DType` and the same valid shape.
    - `validCol` of all tiles must be even (`dst0.GetValidCol() % 2 == 0`). Since all tiles share the same valid shape, this is equivalent to requiring `dst0.GetValidCol() % 2 == 0`.
- **Valid region**:
    - The op uses `dst0.GetValidRow()` / `dst0.GetValidCol()` as the iteration domain; `src0/src1/dst1` are assumed to be compatible.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 64>;
    TileT src0(16, 64), src1(16, 64);
    TileT dst0(16, 64), dst1(16, 64);

    TInterleave(dst1, dst0, src1, src0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
    using TileT = Tile<TileType::Vec, half, 16, 256, BLayout::RowMajor, 16, 256>;
    TileT src0, src1, dst0, dst1;

    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst0, 0x3000);
    TASSIGN(dst1, 0x4000);

    TInterleave(dst1, dst0, src1, src0);
}
```

## Related Instructions

- [TDeInterleave](TDEINTERLEAVE.md) - De-interleave two tiles back into the original even/odd streams (inverse of TInterleave).
