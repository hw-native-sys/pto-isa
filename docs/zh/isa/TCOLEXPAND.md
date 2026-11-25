# TCOLEXPAND

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

![TCOLEXPAND tile operation](../../figures/isa/TCOLEXPAND.svg)

## 简介

将每个源列的第一个元素广播到目标列中。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{0,j} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

- 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。
- **实现检查 (Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品)**：`TileData::DType` 必须为 1、2 或 4 字节类型（b8/b16/b32）：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- **实现检查 (Ascend 950PR/Ascend 950DT)**：`TileData::DType` 必须为 1、2、4 或 8 字节类型（b8/b16/b32/b64）：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`。

- **形状与布局（Ascend 950PR/Ascend 950DT）**：
    - 源和目标均为同类型、非分形 RowMajor Vec Tile。
    - `src.GetValidCol() == dst.GetValidCol()`，源有效行列数必须非零。
    - 只广播源第 0 行，按目标物理行步长写入有效区域。
    - 64 位类型要求物理 `Cols % 4 == 0`，有效列数不必对齐。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```
