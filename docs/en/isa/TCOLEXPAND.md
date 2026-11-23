# TCOLEXPAND

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->



## Tile Operation Diagram

![TCOLEXPAND tile operation](../../figures/isa/TCOLEXPAND.svg)

## Introduction

Broadcast the first element of each source column across the destination column.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{0,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Implementation checks (A2A3)**: `TileData::DType` must be a 1-, 2-, or 4-byte type (b8/b16/b32): `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- **Implementation checks (A5)**: `TileData::DType` must be a 1-, 2-, 4-, or 8-byte type (b8/b16/b32/b64): `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.

- **Shape and layout (A5)**:
    - Source and destination must be non-fractal RowMajor Vec tiles with the same element type.
    - `src.GetValidCol() == dst.GetValidCol()`; source valid dimensions must be nonzero.
    - Only source row 0 is broadcast, using physical destination row strides to write the valid region.
    - 64-bit types require physical `Cols % 4 == 0`; valid columns need not be aligned.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```
