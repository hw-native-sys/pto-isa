# TEXPANDS

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

![TEXPANDS tile operation](../../figures/isa/TEXPANDS.svg)

## Introduction

Broadcast a scalar into a destination tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - For `TileType::Vec` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `bfloat16_t`, `float`.
      - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - For  `TileType::Mat` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `bfloat16_t`, `float`.
      - Static valid bounds: `The range of  TileData::Rows * TileData::Cols * sizeof(T) / 32 is [1, 32767]`.
- **Implementation checks (A5)**:
    - For `TileType::Vec` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
      - Tile layout must be row-major (`TileData::isRowMajor`).
      - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - For  `TileType::Mat` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
      - For`TileDataDst::layout == pto::Layout::NC1HWC0 || TileDataDst::layout == pto::Layout::FRACTAL_Z`:
        - `The range of convtile's (shape0 * shape1 * shape2 * shape3) is [1, 32767]`.
      - For`TileDataDst::layout == pto::Layout::NDC1HWC0 || TileDataDst::layout == pto::Layout::FRACTAL_Z_3D`:
        - `The range of convtile's (shape0 * shape1 * shape2 * shape3 * shape4) is [1, 32767]`.
- **Valid region**:
    - For `TileType::Vec` :
    - The op fills `dst` over `dst.GetValidRow()` / `dst.GetValidCol()`.
    - On A2A3, vector tiles support row-major and column-major layouts with static or dynamic valid dimensions.
    - Valid dimensions may be smaller than the physical tile dimensions; filling follows the physical row or column stride.
    - For  `TileType::Mat` :
    - For Tile : The op fills `dst` over `TileData::Rows` / `TileData::Cols`.
    - For ConvTile : The op fills `dst` over `ConvTileData`'s shape.

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
