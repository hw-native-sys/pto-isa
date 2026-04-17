# pto.taddsc

`pto.taddsc` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

把第一个 tile、一个标量和第二个 tile 融合到一次逐元素加法里。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{scalar} + \mathrm{src1}_{i,j} $$

这条指令的意义在于把“tile + 标量 + tile”的三输入模式直接定义成一条合法操作，避免中间结果再落地一个过渡 tile。

## 语法

同步形式：

```text
%dst = taddsc %src0, %scalar, %src1 : !pto.tile<...>, f32, !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1,
                            WaitEvents&... events);
```

## 输入

- `src0`：第一个源 tile
- `scalar`：广播到所有元素的标量
- `src1`：第二个源 tile
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：融合加法结果 tile

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

当前仓内没有为 `taddsc` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的三输入 tile-标量融合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, out;
  TADDSC(out, a, 2.0f, b);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tlrelu](./tlrelu_zh.md)
- 下一条指令：[pto.tsubsc](./tsubsc_zh.md)
