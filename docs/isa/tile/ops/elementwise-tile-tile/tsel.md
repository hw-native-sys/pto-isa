# TSEL


## Tile Operation Diagram

![TSEL tile operation](../../../../figures/isa/TSEL.svg)

## Introduction

Select between two tiles using a mask tile (per-element selection).

## Math Interpretation

For each element `(i, j)` in the valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true} \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}
$$

## Assembly Syntax

Synchronous form:

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1, TmpTile &tmp, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - `sizeof(TileData::DType)` must be `2` or `4` bytes.
    - `TileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `half` or `bfloat16_t` or `float`.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - The selection domain is `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Implementation checks (A5)**:
    - `sizeof(TileData::DType)` must be `2` or `4` bytes.
    - `TileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `half` or `bfloat16_t` or `float`.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - The selection domain is `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Mask encoding**:
    - The mask tile is interpreted as packed predicate bits in a target-defined layout.

## Temporary Space

### A2A3

`tmp` **is used** as a small buffer to hold the comparison mask (`cmpmask`) copied from the mask tile for each row. The A2A3 implementation uses `set_cmpmask` which requires the mask data to be in a specific UB location.

- `tmp` element type must be `uint32_t`.
- `tmp` size requirement: at least `cmpmaskLen` elements per row, where `cmpmaskLen = 4` for 16-bit data types (`half`, `bfloat16_t`) and `cmpmaskLen = 2` for 32-bit data types (`float`, `int32_t`, `uint32_t`). In bytes, this is always 128 bits (16 bytes).
- A typical `tmp` tile declaration: `Tile<TileType::Vec, uint32_t, 1, 16>` suffices for most use cases.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses vector register-based mask operations (`plds`, `vsel`) and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TSEL(dst, mask, src0, src1, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TASSIGN(mask, 0x4000);
  TASSIGN(tmp,  0x5000);
  TSEL(dst, mask, src0, src1, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
