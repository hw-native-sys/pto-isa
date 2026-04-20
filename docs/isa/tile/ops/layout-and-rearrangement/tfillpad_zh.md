# pto.tfillpad

`pto.tfillpad` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

`TFILLPAD` 复制源 Tile，并把源 valid region 之外的部分用一个编译期确定的 pad 值补满。它最常见的用途是把"运行时有效矩形"扩成"可安全继续参与后续计算的完整静态 Tile"。如果后续操作不想显式处理边界，就需要有人把边界外的位置先变成确定值，`TFILLPAD` 做的正是这件事。

## 机制

设 `VR = src.GetValidRow()` 和 `VC = src.GetValidCol()`。对 `dst` 的每个元素 `(i, j)`：

$$\mathrm{dst}_{i,j} =\begin{cases}\mathrm{src}_{i,j} & \text{当 } i < VR \text{ 且 } j < VC \\ \mathrm{pad} & \text{否则}\end{cases}$$

其中 `pad` 来自 `TileDataDst::PadVal`。常见取值有 `PadValue::Zero`、`PadValue::Min`、`PadValue::Max`，以及通过 `PadValueCustom(...)` 指定的自定义常量。对浮点类型，`Min/Max` 往往会映射到 `-inf/+inf` 一类"适合做极值归约"的值；对整数类型则映射到对应类型的最小值 / 最大值。

## 语法

### PTO-AS

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

相关的同族接口还有 `TFILLPAD_INPLACE(dst, src)` 用于原位填充，以及 `TFILLPAD_EXPAND(dst, src)` 允许 `dst` 比 `src` 更大。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 Tile | 源 Tile |
| `dst` | 输出 Tile | 填充后的目标 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 有效区域内复制源数据、有效区域外用 pad 值填充的 Tile |

## 副作用

`dst` 中有效区域外的内容会被 pad 值覆盖。

## 约束

- Vec Tile 版本要求 `TileDataDst::PadVal != PadValue::Null`
- `src` 和 `dst` 的元素大小必须一致，并且当前实现只接受 `1`、`2` 或 `4` 字节元素
- 如果 `dst.GetValidRow() == 0` 或 `dst.GetValidCol() == 0`，当前 backend 会直接返回，不执行填充
- `TFILLPAD(dst, src)` 和 `TFILLPAD_INPLACE(dst, src)` 要求 `dst.Rows/Cols` 必须与 `src.Rows/Cols` 相同
- `TFILLPAD_EXPAND(dst, src)` 要求 `dst.Rows >= src.Rows` 且 `dst.Cols >= src.Cols`
- 单类型重载 `TFILLPAD(TileData &dst, TileData &src)` 还支持一条 Mat Tile 特化路径，当前只支持 NZ 形态的 Mat Tile（非 row-major，`SLayout::RowMajor`），且 `TileData::PadVal` 为 `PadValue::Zero` 或 `PadValue::Null`

## 异常与非法情形

- 如果 `PadVal` 设置为 `Null` 且使用 Vec Tile 版本，行为未定义
- 如果元素大小不匹配（不是 1、2、4 字节之一），backend 可能报错

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### Vec Tile

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_vec() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16,
                    BLayout::RowMajor, 16, 16, SLayout::NoneBox,
                    TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}
```

### Mat Tile

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mat() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256,
                           BLayout::ColMajor, 1, 224,
                           SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```

## 相关页面

- 指令集总览：[布局与重排](../../layout-and-rearrangement_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
- [Tiles 与有效区域](../../../programming-model/tiles-and-valid-regions_zh.md)
