# pto.tsels

`pto.tsels` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

根据 mask tile，在源 tile 和标量后备值之间逐元素选择。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{当 } \mathrm{mask}_{i,j}\ \text{为真} \\
\mathrm{scalar} & \text{否则}
\end{cases}
$$

这里的 `mask` tile 采用目标定义的 packed predicate 编码；`tmp` 是为谓词展开 / 中间处理准备的临时 tile。

## 语法

同步形式：

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

- `mask`：谓词 tile；为真时从 `src` 取值，否则从 `scalar` 取值
- `src`：源 tile
- `scalar`：后备标量值
- `tmp`：谓词展开所需的临时 tile
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素选择后的结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- `mask` tile 采用目标定义的 packed predicate 编码。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `sizeof(TileDataDst::DType)` 必须是 2 或 4 字节
- 支持类型：`half`、`float16_t`、`float`、`float32_t`
- `dst` 与 `src` 必须使用相同元素类型
- `dst` 与 `src` 必须是行主序
- 运行时要求：`src.GetValidRow()/GetValidCol()` 必须与 `dst.GetValidRow()/GetValidCol()` 一致

### A5

- `sizeof(TileDataDst::DType)` 可以是 1、2 或 4 字节
- 支持类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`float`
- `dst` 与 `src` 必须使用相同元素类型
- `dst`、`mask` 与 `src` 都必须是行主序
- 运行时要求：`src.GetValidRow()/GetValidCol()` 必须与 `dst.GetValidRow()/GetValidCol()` 一致

## 性能

当前仓内没有把 `tsels` 单独落成公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile 选择路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TSELS(dst, mask, src, tmp, scalar);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tcmps](./tcmps_zh.md)
- 下一条指令：[pto.tmins](./tmins_zh.md)
