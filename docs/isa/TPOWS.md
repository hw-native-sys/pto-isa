# TPOWS


## Tile Operation Diagram

![TPOWS tile operation](../figures/isa/TPOWS.svg)

## Introduction

Elementwise power operation with scalar exponent: computes `base` raised to a scalar power `exp` for each element.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{base}_{i,j}^{\mathrm{exp}} $$

For floating-point types, the computation follows: `dst = exp(ln(|base|) * exp)` with special case handling for negative base values and integer exponents.

## Assembly Syntax

Synchronous form:

```text
%dst = tpows %base, %exp, %tmp : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```text
%dst = pto.tpows %base, %exp, %tmp : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpows ins(%base, %exp, %tmp : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
          typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOWS(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp,
                               WaitEvents &... events);
```

`PrecisionType` has the following values available:

* `PowAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `PowAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower.

## Constraints

### General Constraints

- `dst` and `base` must both be `TileType::Vec`.
- Both tiles must use row-major layout (`TileData::isRowMajor`).
- `dst` and `base` must have the same element type.
- Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- Runtime valid region checks:
    - `dst.GetValidRow() == base.GetValidRow()`
    - `dst.GetValidCol() == base.GetValidCol()`
- The intrinsic signature requires an explicit `tmp` operand.

### A2A3 Implementation Checks

- Supported element types: `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `uint16_t`, `uint8_t`, `float`.
- `HIGH_PRECISION` algorithm is not supported on A2A3; `PrecisionType` option is ignored.
- Additional runtime valid region checks for `tmp`:
    - `dst.GetValidRow() == tmp.GetValidRow()`
    - `dst.GetValidCol() == tmp.GetValidCol()`

### A5 Implementation Checks

- For `DEFAULT` algorithm: supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`, `bfloat16_t`.
- For `HIGH_PRECISION` algorithm: supported element types are `half`, `float`, `bfloat16_t` (floating-point only).
- Integer types use a separate integer power computation path.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT base, dst, tmp;
  TPOWS(dst, base, 2.0f, tmp);
  TPOWS<PowAlgorithm::HIGH_PRECISION>(dst, base, 2.0f, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT base, dst, tmp;
  TASSIGN(base, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TPOWS(dst, base, 2.0f, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpows %base, %exp, %tmp : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpows %base, %exp, %tmp : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpows %base, %exp, %tmp : !pto.tile<...>, dtype
# AS Level 2 (DPS)
pto.tpows ins(%base, %exp, %tmp : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```