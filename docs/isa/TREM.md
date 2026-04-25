# TREM


## Tile Operation Diagram

![TREM tile operation](../figures/isa/TREM.svg)

## Introduction

Elementwise remainder of two tiles. The result has the same sign as the divider.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \bmod \mathrm{src1}_{i,j}$$

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = trem %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- **Implementation Checks (A2A3)**:
    - `dst`, `src0`, and `src1` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst`, `src0`, and `src1` must be vector tiles.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow() > 0` and `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol() > 0`.
    - **tmp Buffer Requirements**:
      - `tmp.GetValidCol() >= dst.GetValidCol()` (at least as many columns as dst)
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - Data type must match `TileDataDst::DType`.
- **Implementation Checks (A5)**:
    - `dst`, `src0`, and `src1` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst`, `src0`, and `src1` must be vector tiles.
    - Static valid bounds: `ValidRow <= Rows` and `ValidCol <= Cols` for all tiles.
    - Runtime: `dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow()` and `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol()`.
    - Note: tmp parameter is accepted but not validated or used on A5.
- **Division by Zero**:
    - Behavior is target-defined; the CPU simulator asserts in debug builds.
- **Valid Region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
- **For `int32_t` Inputs (A2A3 Only)**: Both `src0` and `src1` elements must be in the range `[-2^24, 2^24]` (i.e., `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  Tile<TileType::Vec, int32_t, 16, 16> tmp;
  TREM(out, a, b, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trem %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

