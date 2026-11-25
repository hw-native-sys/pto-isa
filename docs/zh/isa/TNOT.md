# TNOT

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

![TNOT tile operation](../../figures/isa/TNOT.svg)

## 简介

Tile的逐元素按位取反。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \sim\mathrm{src}_{i,j} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `TileData::DType` 必须是以下之一： `int16_t`， `uint16_t`， `uint32_t`， `int32_t`， `int8_t`， `uint8_t`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 静态有效边界： `TileData::ValidRow <= TileData::Rows`且`TileData::ValidCol <= TileData::Cols`。
    - 运行时： `src`且`dst` tiles应具有相同的 `validRow/validCol`。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` 必须是以下之一： `uint32_t`， `int32_t`， `int64_t`， `uint64_t`， `uint16_t`， `int16_t`， `uint8_t`，  `int8_t`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 静态有效边界： `TileData::ValidRow <= TileData::Rows`且`TileData::ValidCol <= TileData::Cols`。
    - 运行时： `src`且`dst` tiles应具有相同的 `validRow/validCol`。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域； `src/dst` 假定是兼容的 （此操作中不通过显式运行时检查进行验证）.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT x, out;
  TNOT(out, x);
}
```
