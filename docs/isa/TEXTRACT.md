# TEXTRACT


## Tile Operation Diagram

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## Introduction

Extract a sub-tile from a source tile.

## Math Interpretation

Conceptually copies a window starting at `(indexRow, indexCol)` from `src` into `dst`. Exact mapping depends on layouts.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                              WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `DstTileData::DType` must equal `SrcTileData::DType` and must be one of: `int8_t`, `half`, `bfloat16_t`, `float`.
  - Source fractal must satisfy: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`.
  - Runtime bounds checks:
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`
  - Destination must be `TileType::Left` or `TileType::Right` with a target-supported fractal configuration.
- **Implementation checks (A5)**:
  - `DstTileData::DType` must equal `SrcTileData::DType` and must be one of: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`.
  - Source fractal must satisfy: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)` for Left/Right, `(SFractal == RowMajor && isRowMajor)` for ScaleLeft, `(SFractal == ColMajor && !isRowMajor)` for ScaleRight.
  - Destination supports `Mat -> Left/Right/Scale` and also supports `Vec -> Mat` for specific tile locations (no explicit runtime bounds assertions are enforced in `TEXTRACT_IMPL` on this target).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

