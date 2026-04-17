# pto.tlog

`pto.tlog` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素自然对数。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \log(\mathrm{src}_{i,j}) $$

它是 tile 路径上的一元对数操作，用于归一化、损失计算前处理和指数域反变换。

## 语法

### PTO-AS

```text
%dst = tlog %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = LogAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

可选 `PrecisionType`：

- `LogAlgorithm::DEFAULT`
- `LogAlgorithm::HIGH_PRECISION`

## 输入

- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素对数结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 支持类型当前是 `float` / `half`
- tile 必须是行主序向量 tile
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定
- 对 `log(<=0)` 的域外情况，行为由目标 profile 定义
- 高精度算法只在 A5 有效

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### NPU

- 支持类型：`float`、`half`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`

## 性能

当前仓内没有把 `tlog` 单列成 tile 公开 cost bucket，但它显然属于一元超越函数路径；在 A2/A3 上通常会比普通二元算术更贵。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLOG(out, x);
  TLOG<LogAlgorithm::HIGH_PRECISION>(out, x);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
