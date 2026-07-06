# TPOW


## Tile Operation Diagram

![TPOW tile operation](../figures/isa/TPOW.svg)

## Introduction

Elementwise power operation: computes `base` raised to the power of `exp` for each element.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{base}_{i,j}^{\mathrm{exp}_{i,j}} $$

For floating-point types, the computation follows: `dst = exp(ln(|base|) * exp)` with special case handling for negative base values and integer exponents.

## Assembly Syntax

Synchronous form:

```text
%dst = tpow %base, %exp, %tmp : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpow %base, %exp, %tmp : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpow ins(%base, %exp, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile,
          typename TmpTile, typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOW(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp, WaitEvents &... events);
```

`PrecisionType` has the following values available:

* `PowAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `PowAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower.

## Constraints

### General Constraints

- `dst`, `base`, and `exp` must all be `TileType::Vec`.
- All tiles must use row-major layout (`TileData::isRowMajor`).
- `dst`, `base`, and `exp` must have the same element type.
- Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- Runtime valid region checks:
    - `dst.GetValidRow() == base.GetValidRow()`
    - `dst.GetValidCol() == base.GetValidCol()`
    - `dst.GetValidRow() == exp.GetValidRow()`
    - `dst.GetValidCol() == exp.GetValidCol()`
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

## Temporary Space

### A2A3

- For **floating-point** types (`float`): `tmp` **is used** as intermediate scratch storage. The implementation computes `pow(base, exp) = exp(ln(|base|) * exp)` and uses `tmp` to store the `exp(ln(|base|) * exp)` result with absolute value for special-case handling (negative base with odd integer exponent).
  - `tmp` must have the same element type as `dst`/`base`/`exp`.
  - `tmp.GetValidRow() >= dst.GetValidRow()` and `tmp.GetValidCol() >= dst.GetValidCol()`.
- For **integer** types: `tmp` is **not used**. The integer power path uses scalar computation without scratch tile storage.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses vector register-based computation and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT base, exp, dst, tmp;
  TPOW(dst, base, exp, tmp);
  TPOW<PowAlgorithm::HIGH_PRECISION>(dst, base, exp, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT base, exp, dst, tmp;
  TASSIGN(base, 0x1000);
  TASSIGN(exp, 0x2000);
  TASSIGN(dst, 0x3000);
  TASSIGN(tmp, 0x4000);
  TPOW(dst, base, exp, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpow %base, %exp, %tmp : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dst = pto.tpow %base, %exp, %tmp : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpow %base, %exp, %tmp : !pto.tile<...>
# AS Level 2 (DPS)
pto.tpow ins(%base, %exp, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```