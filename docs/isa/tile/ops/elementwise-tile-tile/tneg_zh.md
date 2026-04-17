# pto.tneg

`pto.tneg` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素取负。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = -\mathrm{src}_{i,j} $$

这是一元算术操作，用来完成符号翻转或构造相反数。

## 语法

### PTO-AS

```text
%dst = tneg %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tneg %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入

- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素取负结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

- `pto.tneg` 在 CPU 仿真、A2/A3 和 A5 上保留一致的 PTO 可见语义，但支持子集仍取决于 profile。

## 性能

### A2A3

英文页当前把 `TNEG` 归到一元 tile 运算桶：

| 指标 | 数值 |
| --- | --- |
| 启动时延 | 13 |
| 完成时延 | 26 |
| 每次 repeat 吞吐 | 1 |
| 流水间隔 | 18 |

### A5

当前手册未单列 `tneg` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TNEG(out, x);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.trelu](./trelu_zh.md)
- 下一条指令：[pto.trem](./trem_zh.md)
