# TMAX


## Tile Operation Diagram

![TMAX tile operation](../figures/isa/TMAX.svg)

## Introduction

Elementwise maximum of two tiles.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = tmax %src0, %src1 : !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**:
  - `TileData::DType` must be one of: `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`,  `int8_t`, `float`, `half`.
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Valid region**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain; `src0/src1` are assumed to be compatible (not validated by explicit runtime checks in this op).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TMAX(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TMAX(dst, src0, src1);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tmax %src0, %src1 : !pto.tile<...>
# IR Level 2 (DPS)
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

