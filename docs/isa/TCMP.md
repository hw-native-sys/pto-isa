# TCMP


## Tile Operation Diagram

![TCMP tile operation](../figures/isa/TCMP.svg)

## Introduction

Compare two tiles and write a packed predicate mask.

## Math Interpretation

Conceptually, for each element `(i, j)` in the valid region, define a predicate:

$$ p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

The predicate mask is stored in `dst` using an implementation-defined packed layout.

## Assembly Syntax

Synchronous form:

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode cmpMode, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - Input type must be one of: `int32_t`, `half`, `float`.
    - Output type must be `uint8_t`.
    - `src0/src1/dst` tile location must be `TileType::Vec`.
    - Static valid bounds: `TileDataSrc::ValidRow <= TileDataSrc::Rows` and `TileDataSrc::ValidCol <= TileDataSrc::Cols`.
    - Runtime: `src0` and `src1` must have equal valid row and column counts, and `src0.GetValidRow() == dst.GetValidRow()`.
    - Destination valid columns describe packed capacity and need not equal source valid columns.
    - For `int32_t` input, `EQ` and `NE` are supported. `NE` inverts the equality result; other modes use the `EQ` path.
- **Implementation checks (A5)**:
    - Input type must be one of: `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `uint16_t`, `int16_t`, `uint8_t`, `int8_t`, `float`, `half`, `bfloat16_t`.
    - Output is packed predicate bytes; a RowMajor `uint8_t` mask tile may be used.
    - Implemented (see `include/pto/npu/a5/TCmp.hpp`).
    - The iteration domain is `src0.GetValidRow()` / `src0.GetValidCol()`; `src1` must provide the corresponding valid elements. Destination valid columns describe packed capacity, not the comparison count.
- **Mask encoding**:
    - For 64-bit input (A5), the predicate for column `j` occupies bit `j % 8` of byte `j / 8` in that row, least significant bit first.
    - A `uint8_t` mask may use valid shape `[R, ceil(C / 8)]` with physical `Cols` aligned to 32 bytes, where `[R,C]` is the source valid shape. Rows use the physical destination stride; padding after the last valid bit is unspecified.
    - The mask tile is interpreted as packed predicate bits in a target-defined layout.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
