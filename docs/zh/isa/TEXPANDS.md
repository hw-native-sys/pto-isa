# TEXPANDS

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 指令示意图

![TEXPANDS tile operation](../../figures/isa/TEXPANDS.svg)

## 简介

将标量广播到目标Tile中。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 对于Tile位置是向量（`TileData::Loc == TileType::Vec`）:
    - `TileData::DType` 必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
    - 静态有效边界： `TileData::ValidRow <= TileData::Rows`且`TileData::ValidCol <= TileData::Cols`。
    - 对于Tile位置是Mat（`TileData::Loc == TileType::Mat`）:
    - `TileData::DType` 必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
    - 有效边界：`TileData::Rows * TileData::Cols * sizeof(TileData::DType) / 32` 必须在`[1, 32767]`范围内。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 对于Tile位置是向量（`TileData::Loc == TileType::Vec`）:
    - 静态有效边界： `TileData::ValidRow <= TileData::Rows`且`TileData::ValidCol <= TileData::Cols`。
    - `TileData::DType` 必须是以下之一： `uint8_t`， `int8_t`， `uint16_t`， `int16_t`， `uint32_t`， `int32_t`， `int64_t`， `uint64_t`， `half`， `bfloat16_t`， `float`。
    - 对于Tile位置是Mat（`TileData::Loc == TileType::Mat`）:
    - `TileData::DType` 必须是以下之一： `uint8_t`， `int8_t`， `uint16_t`， `int16_t`， `uint32_t`， `int32_t`， `int64_t`， `uint64_t`， `half`， `bfloat16_t`， `float`。
    - 对于`TileData::layout == pto::Layout::NC1HWC0 || TileData::layout == pto::Layout::FRACTAL_Z`：
      - `TileData::shape0 * TileData::shape1 * TileData::shape2 * TileData::shape3` 必须在`[1, 32767]`范围内。
    - 对于`TileData::layout == pto::Layout::NDC1HWC0 || TileData::layout == pto::Layout::FRACTAL_Z_3D`：
      - `TileData::shape0 * TileData::shape1 * TileData::shape2 * TileData::shape3 * TileData::shape4` 必须在`[1, 32767]`范围内。
- **有效区域**:
    - 对于Tile位置是向量（`TileData::Loc == TileType::Vec`）:
    - 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上填充 `dst`。
    - 在 A2/A3 上，向量 Tile 支持行主序和列主序布局，有效尺寸可以是静态或动态的。
    - 有效尺寸可以小于 Tile 的物理尺寸；填充时使用物理行步长或列步长。
    - 对于Tile位置是Mat（`TileData::Loc == TileType::Mat`）:
    - 对于Tile，该操作在 `TileData::Rows` / `TileData::Cols` 上填充 `dst`。
    - 对于convTile，该操作在`ConvTileData`的`shape`内填充`dst`。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TEXPANDS(dst, 0.0f);
}
```

### 手动（Manual）

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
