# pto.trsqrt

`pto.trsqrt` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素倒数平方根。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \frac{1}{\sqrt{\mathrm{src}_{i,j}}} $$

它常用于范数归一化、标准化和一类需要快速 `1/sqrt(x)` 的数值路径。

## 语法

### PTO-AS

```text
%dst = trsqrt %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `%src`：源 tile
- `%tmp`：可选临时 tile；提供时通常触发高精度路径
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素倒数平方根结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 支持类型当前是 `float` / `half`
- tile 必须是行主序向量 tile
- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定
- 对 `src == 0` 或负输入这类情况，行为由目标 profile 定义
- 当 `tmp` 提供时，NPU 路径要求它至少有 32 字节，以触发高精度实现

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### NPU

- `tmp` 至少需要 32 字节；提供后会使用高精度版本
- 支持类型：`float`、`half`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`

## 性能

### A2A3

英文页当前把 `TRSQRT` 归到一元超越函数桶：

| 指标 | 数值 |
| --- | --- |
| 启动时延 | 13 |
| 完成时延 | 26 |
| 每次 repeat 吞吐 | 1 |
| 流水间隔 | 18 |

### A5

当前手册未单列 `trsqrt` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TRSQRT(dst, src);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tsel](./tsel_zh.md)
- 下一条指令：[pto.tsqrt](./tsqrt_zh.md)
