# TABS


## Tile Operation Diagram

![TABS tile operation](../figures/isa/TABS.svg)

## Introduction

Elementwise absolute value of a tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileData& dst, TileData& src, WaitEvents&... events);
```

## Constraints

- **Implementation checks (CPU sim)**:
  - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `half`, `float`.
  - The implementation iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Implementation checks (NPU)**:
  - `TileData::DType` must be one of: `float` or `half`;
  - Tile location must be vector (`TileData::Loc == TileType::Vec`);
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`;
  - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`;
  - Tile layout must be row-major (`TileData::isRowMajor`).
- **Valid region**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TABS(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TABS(dst, src);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tabs %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

