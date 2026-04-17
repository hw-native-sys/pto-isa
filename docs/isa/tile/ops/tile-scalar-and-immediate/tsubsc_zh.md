# pto.tsubsc

`pto.tsubsc` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

把第一个 tile、一个标量和第二个 tile 融合到一次逐元素减法 / 加法组合里。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{scalar} + \mathrm{src1}_{i,j} $$

和 `taddsc` 一样，它把三输入融合模式直接暴露成单条 tile 操作。

## 语法

同步形式：

```text
%dst = tsubsc %src0, %scalar, %src1 : !pto.tile<...>, f32, !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tsubsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1,
                            WaitEvents&... events);
```

## 输入

- `src0`：第一个源 tile
- `scalar`：广播到所有元素的标量
- `src1`：第二个源 tile
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：融合结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- tile 位置必须是向量 tile。
- 静态 valid 边界必须合法。
- 运行时要求：`dst`、`src0`、`src1` 的 valid row / valid col 一致。
- 标量类型必须匹配 tile 数据类型。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`int16_t`、`half`、`float`
- tile 布局必须是行主序

### A5

- `TileData::DType` 必须属于：`int32_t`、`int16_t`、`half`、`float`
- tile 布局必须是行主序

## 性能

当前仓内没有为 `tsubsc` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的三输入 tile-标量融合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, out;
  TSUBSC(out, a, 2.0f, b);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.taddsc](./taddsc_zh.md)
