# TNOT


## Tile Operation Diagram

![TNOT tile operation](../figures/isa/TNOT.svg)

## Introduction

Elementwise bitwise NOT of a tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \sim\mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = tnot %src : !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileData& dst, TileData& src, WaitEvents&... events);
```

## Constraints

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::DType` must be one of: `int16_t`, `uint16_t`.
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**:
  - `TileData::DType` must be one of: `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`,  `int8_t`.
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Valid region**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain; `src/dst` are assumed to be compatible (not validated by explicit runtime checks in this op).


## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT x, out;
  TNOT(out, x);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tnot %src : !pto.tile<...>
# IR Level 2 (DPS)
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

