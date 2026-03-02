# TEXPANDS


## Tile Operation Diagram

![TEXPANDS tile operation](../figures/isa/TEXPANDS.svg)

## Introduction

Broadcast a scalar into a destination tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = texpands %scalar : f32, !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData& dst, typename TileData::DType scalar, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- **Implementation checks (A5)**:
  - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Tile layout must be row-major (`TileData::isRowMajor`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- **Valid region**:
  - The op fills `dst` over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TEXPANDS(dst, 0.0f);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TEXPANDS(dst, 0.0f);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = texpands %scalar : f32, !pto.tile<...>
# IR Level 2 (DPS)
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

