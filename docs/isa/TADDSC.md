# TADDSC


## Tile Operation Diagram

![TADDSC tile operation](../figures/isa/TADDSC.svg)

## Introduction

Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{scalar} + \mathrm{src1}_{i,j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = taddsc %src0, %scalar, %src1 : !pto.tile<...>, f32, !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1,
                            WaitEvents&... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, out;
  TADDSC(out, a, 2.0f, b);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = taddsc %src0, %scalar, %src1 : !pto.tile<...>, f32, !pto.tile<...>
# IR Level 2 (DPS)
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

