# pto.trecip

`pto.trecip` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素倒数。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \frac{1}{\mathrm{src}_{i,j}} $$

它适合在后续仍要与其他 tile 做乘法组合时，替代显式除法。

## 语法

### PTO-AS

```text
%dst = trecip %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

可选 `PrecisionType`：

- `RecipAlgorithm::DEFAULT`
- `RecipAlgorithm::HIGH_PRECISION`

## 输入

- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素倒数结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标定义；CPU 模拟器在调试构建下会断言。
- 高精度算法只在 A5 有效。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### NPU

- 支持类型：`float`、`half`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`
- A3 不支持源 tile 与目标 tile 绑定到同一片内存

## 性能

当前仓内没有为 `trecip` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的一元数学路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRECIP(out, x);
  TRECIP<RecipAlgorithm::HIGH_PRECISION>(out, x);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.txor](./txor_zh.md)
- 下一条指令：[pto.tprelu](./tprelu_zh.md)
