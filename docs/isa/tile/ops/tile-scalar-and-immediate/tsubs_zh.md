# pto.tsubs

`pto.tsubs` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

把一个标量逐元素从 tile 上减去。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar} $$

它和 `tadds` 一样，作用对象是 tile payload，标量会广播到整个 valid region。

## 语法

同步形式：

```text
%dst = tsubs %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素减法后的结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src0` 必须使用相同元素类型。
- 标量类型必须匹配 `TileDataSrc::DType`。
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`

### A5

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`

## 性能

当前仓内没有把 `tsubs` 单独落成公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile-标量减法路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TSUBS(out, x, 1.0f);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tadds](./tadds_zh.md)
- 下一条指令：[pto.tdivs](./tdivs_zh.md)
