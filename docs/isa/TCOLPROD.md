# TCOLPROD


## Tile Operation Diagram

![TCOLPROD tile operation](../figures/isa/TCOLPROD.svg)

## Introduction

Reduce each column by multiplying across rows.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \prod_{i=0}^{R-1} \mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut& dst, TileDataIn& src, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `dst` and `src` must be `TileType::Vec`.
- Tile layout: both tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- DType consistency: `dst.DType == src.DType`.
- Supported `src.DType`:
  - A2A3: `half`, `float`, `int16_t`, `int32_t`.
  - A5: `half`, `float`, `bfloat16`, `int16_t`, `int32_t`, `uint16_t`, `uint32_t`.
- Runtime valid checks:
  - `src.GetValidCol() == dst.GetValidCol()`.
  - If `src.GetValidRow() == 0` or `src.GetValidCol() == 0`, the implementation returns early.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TCOLPROD(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCOLPROD(dst, src);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolprod %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

