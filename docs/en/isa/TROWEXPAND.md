# TROWEXPAND

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

![TROWEXPAND tile operation](../../figures/isa/TROWEXPAND.svg)

## Introduction

Broadcast the first element of each source row across the destination row.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,0} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

Implementation checks (NPU):

- Tile Type: `dst` and `src` must be `TileType::Vec`.
- Tile layout: non-fractal RowMajor `dst`; non-fractal RowMajor or ColMajor `src` (`SLayout::NoneBox`). Input and output element types must match.
- A5 requires source static `ValidCol` to be 1 or dynamic (-1); use valid column count 1 for scalar row broadcast.
- Data type (A2A3): element types must be one of: `int8_t` or `uint8_t` or `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `half` or `bfloat16_t` or `float`.
- Data type (A5): element types must be one of: `int8_t` or `uint8_t` or `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `int64_t` or `uint64_t` or `half` or `bfloat16_t` or `float`.
- Runtime valid checks:
    - A2A3: returns early if any of `dstValidRow`, `dstValidCol`, `srcValidRow`, `srcValidCol` is zero.
    - A5: asserts `srcValidRow == dstValidRow` and asserts `srcValidRow != 0 && srcValidCol != 0`.

- 64-bit input (A5): an ND source may use physical `[64,4]`, valid `[64,1]`; a DN source may use compact `[64,1]`. Each row broadcasts `src[i,0]` using the physical `RowStride`.
- Unlike binary row-broadcast instructions such as [TROWEXPANDADD](TROWEXPANDADD.md), this instruction uses one scalar per row for ND sources as well, without repeating a 32-byte block.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TROWEXPAND(dst, src);
}
```
