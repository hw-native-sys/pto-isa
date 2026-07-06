# TXOR


## Tile Operation Diagram

![TXOR tile operation](../figures/isa/TXOR.svg)

## Introduction

Elementwise bitwise XOR of two tiles.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \oplus \mathrm{src1}_{i,j} $$

## Assembly Syntax

Synchronous form:

```text
%dst = txor %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Implementation checks (A5)**:
    - `dst`, `src0`, and `src1` element types must match.
    - Supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`.
    - `dst`, `src0`, and `src1` must be row-major.
    - `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must match `dst`.
- **Implementation checks (A2A3)**:
    - `dst`, `src0`, `src1`, and `tmp` element types must match.
    - Supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`.
    - `dst`, `src0`, `src1`, and `tmp` must be row-major.
    - `src0`, `src1`, and `tmp` valid shapes must match `dst`.
    - In manual mode, `dst`, `src0`, `src1`, and `tmp` must not overlap in memory.

## Temporary Space

### A2A3

`tmp` **is used** as intermediate scratch storage. The A2A3 implementation computes XOR via decomposition: `XOR(a,b) = AND(NOT(AND(a,b)), OR(a,b))`, which requires `tmp` to hold the intermediate `OR(a,b)` result.

- `tmp` must have the same element type as `dst`/`src0`/`src1`.
- `tmp` must be row-major.
- `tmp.GetValidRow() >= dst.GetValidRow()` and `tmp.GetValidCol() >= dst.GetValidCol()`.
- In manual mode, `tmp` must not overlap in memory with `dst`, `src0`, or `src1`.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses the `vxor` vector instruction directly and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc0 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc1 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc0 src0;
  TileSrc1 src1;
  TileTmp tmp;
  TXOR(dst, src0, src1, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = txor %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

