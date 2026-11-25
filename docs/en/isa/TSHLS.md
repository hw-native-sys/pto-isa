# TSHLS

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

![TSHLS tile operation](../../figures/isa/TSHLS.svg)

## Introduction

Elementwise shift-left of a tile, shift bits given by scalar.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - Supported element types are `int32_t`, `int`, `int16_t`, `uint32_t`, `unsigned int`, and `uint16_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - Scalar only supports zero and positive values.
- **Implementation checks (A5)**:
    - Supported element types are `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, and `uint8_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Static valid bounds must satisfy `ValidRow <= Rows` and `ValidCol <= Cols` for both tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - Scalar only supports zero and positive values.
    - For `int64_t` and `uint64_t`, the effective shift amount is `scalar & 63`.
- **Implementation checks (A6)**:
    - Supported element types are `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, and `uint8_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Static valid bounds must satisfy `ValidRow <= Rows` and `ValidCol <= Cols` for both tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - The scalar is interpreted as signed: a negative amount reverses the shift direction (right shift by `|scalar|`).
    - For `int64_t` and `uint64_t`, the effective shift amount is `scalar & 63`.
- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TSHLS(dst, src, 0x2);
}
```
