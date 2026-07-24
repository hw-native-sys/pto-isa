# TCONCAT

## Tile Operation Diagram

### Basic Form (3 arguments)

![TCONCAT basic form](../figures/isa/TCONCAT.svg)

### Indexed Form (5-6 arguments)

![TCONCAT indexed form](../figures/isa/TCONCAT_idx.svg)

## Introduction

Concatenate two source tiles (`src0` and `src1`) horizontally into a destination tile (`dst`) along the column dimension. Each row of `dst` contains the concatenation of corresponding rows from `src0` and `src1`.

`TCONCAT` is used for:

- Concatenating two tiles along the column axis (horizontal concatenation)
- Joining tiles in attention and transformer architectures (e.g., concatenating KV cache entries)
- Combining partial results from split operations

## Math Interpretation

For each row `i` in the valid region:

$$ \mathrm{dst}_{i, j} = \begin{cases} \mathrm{src0}_{i, j} & \text{if } 0 \le j < \mathrm{validCols0} \\ \mathrm{src1}_{i, j - \mathrm{validCols0}} & \text{if } \mathrm{validCols0} \le j < \mathrm{validCols0} + \mathrm{validCols1} \end{cases} $$

Where `validCols0 = src0.GetValidCol()` and `validCols1 = src1.GetValidCol()`.

## Assembly Syntax

### AS Level 1 (SSA)

```text
%dst = pto.tconcat %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tconcat ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCONCAT(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, WaitEvents &... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TCONCAT(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx, WaitEvents &... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileDstIdx, typename TileSrc0Idx, typename TileSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TCONCAT(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileDstIdx &dstIdx, TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx, WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `TCONCAT` has three overload variants:
    - basic form: `TCONCAT(dst, src0, src1)` - concatenates full valid regions
    - indexed form (5 args): `TCONCAT(dst, src0, src1, src0Idx, src1Idx)` - uses per-row index tiles to specify dynamic column counts
    - indexed form (6 args): `TCONCAT(dst, src0, src1, dstIdx, src0Idx, src1Idx)` - also outputs the concatenated column count per row
- All tiles must have `TileType::Vec` (vector tiles)
- All tiles must use row-major layout (`isRowMajor == true`)

### Shape constraints

- Basic form:
    - `dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow()`
    - A2A3: `src0.GetValidCol() + src1.GetValidCol() <= TileDataDst::Cols` (total columns must not exceed dst physical capacity)
    - A5: `dst.GetValidCol() == src0.GetValidCol() + src1.GetValidCol()` (total columns must equal dst valid columns)
- Indexed form:
    - Same row count constraints as basic form
    - Column counts are determined dynamically from index tiles
    - `dstIdx.GetValidRow() == 1` for 6-argument form

### Data type constraints

- Supported element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`
- Source and destination tiles must have identical element type
- Index tiles must use integer types (`int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`)

### A5 implementation checks

- All tiles must be `TileType::Vec`
- All tiles must be row-major layout
- `validRows` must not exceed the physical tile rows for any operand
- Index tiles (if provided) must satisfy type compatibility checks

## Examples

### Auto Mode

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 32>;
    TileT src0(16, 16);
    TileT src1(16, 16);
    TileT dst(16, 32);

    TCONCAT(dst, src0, src1);
}
```

### Manual Mode

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
    using TileT = Tile<TileType::Vec, half, 16, 64, BLayout::RowMajor, 16, 64>;
    TileT src0, src1, dst;

    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst, 0x3000);

    src0.SetValidRegion(16, 32);
    src1.SetValidRegion(16, 32);

    TCONCAT(dst, src0, src1);
}
```

### Indexed Form Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_indexed() {
    using TileT = Tile<TileType::Vec, float, 16, 64>;
    using IdxTileT = Tile<TileType::Vec, int32_t, 16, 1>;

    TileT src0(16, 32);
    TileT src1(16, 32);
    TileT dst(16, 64);
    IdxTileT src0Idx, src1Idx;

    TCONCAT(dst, src0, src1, src0Idx, src1Idx);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tconcat %src0, %src1 : (!pto.tile<16x32xf32>, !pto.tile<16x32xf32>) -> !pto.tile<16x64xf32>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %src0, @tile(0x1000)
# pto.tassign %src1, @tile(0x2000)
# pto.tassign %dst, @tile(0x3000)
%dst = pto.tconcat %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## Related Instructions

- [TINSERT](TINSERT.md) - Insert a sub-tile into a destination tile at specified offset
- [TEXTRACT](TEXTRACT.md) - Extract a sub-tile from a source tile
- [TRESHAPE](TRESHAPE.md) - Reinterpret a tile as another tile type/shape
