# pto.tmaxs

`pto.tmaxs` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

对 tile 和标量逐元素取最大值。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

标量会广播到整个 valid region。

## 语法

同步形式：

```text
%dst = tmaxs %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMAXS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素最大值结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- tile 位置必须是向量 tile
- 标量类型必须与 tile 数据类型匹配
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`int16_t`、`half`、`float`
- tile 布局必须是行主序

### A5

- `TileData::DType` 必须属于：`int32_t`、`uint32_t`、`float`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`uint8_t`、`int8_t`
- tile 布局必须是行主序

## 性能

当前仓内虽然存在 `TMAXS_IMPL`，但公开 costmodel 没有像 `TADDS` / `TMULS` 那样直接单列 `TMAXS` 常量表。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile max 路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TMAXS(out, x, 0.0f);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.trems](./trems_zh.md)
- 下一条指令：[pto.tands](./tands_zh.md)
