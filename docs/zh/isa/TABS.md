# TABS

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

![TABS tile operation](../../figures/isa/TABS.svg)

## 简介

Tile的逐元素绝对值。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

- **实现检查 (CPU sim)**:
    - `dst` 和 `src` 的数据类型必须一致，且必须是以下之一：`int32_t`、`int`、`int16_t`、`half`、`float`。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`，不一致会触发断言。
    - 实现在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。
- **实现检查 (Costmodel)**:
    - `TileData::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`uint8_t`、`half`、`float`。
- **实现检查 (NPU)**:
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品 `TileData::DType` 必须是以下之一：`float` 或 `half`。
    - Ascend 950PR/Ascend 950DT `TileData::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`int64_t`、`half`、`float`。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
    - 在元素类型一致的前提下，`dst` 和 `src` 可以使用不同的 C++ Tile 类型，包括混用静态和动态 `ValidRow`/`ValidCol` 模板参数。CPU_SIM 按每个操作数自身的 Tile 布局和物理形状计算地址。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TABS(dst, src);
}
```

### 自动（混用静态和动态有效形状）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mixed_valid_shape() {
  using DynamicTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, -1, -1>;
  using StaticTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, 16, 16>;
  DynamicTile dst(16, 16);
  StaticTile src;
  TABS(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TABS(dst, src);
}
```
